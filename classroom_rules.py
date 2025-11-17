# classroom_rules.py
"""
Rule-based classroom scene understanding WITH emotion.

Input:
  - frame (BGR image)
  - YOLO detections: list of {"class", "box", "conf"}

Output:
  - state_id: int
      0 = normal class activity
      1 = lone child may need attention
      2 = dangerous object near child
      3 = empty classroom
  - state_text: short label
  - reasons: explanation (includes emotion if available)
"""

import math
from typing import List, Dict, Tuple, Set

try:
    from deepface import DeepFace
    HAVE_DEEPFACE = True
except Exception:
    HAVE_DEEPFACE = False
    print("[WARN] DeepFace not available – emotion defaults to neutral.")

# Obviously dangerous things in a classroom (hardcoded for speed)
OBVIOUS_DANGEROUS_CLASSES = {"knife", "scissors", "fork", "bottle", "gun", "fire", "lighter"}

# Cache for LLM-classified dangerous objects (updated dynamically)
_llm_dangerous_cache: Set[str] = set()

NEGATIVE_EMOTIONS = {"sad", "fear", "angry", "disgust"}
POSITIVE_EMOTIONS = {"happy", "surprise"}


def _center(box: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _get_child_emotion(frame, person_boxes):
    """
    Use first person as main child and estimate emotion.
    Returns (emotion_label, extra_info).
    If DeepFace not available or fails -> ('neutral', 'Emotion not available or defaulted').
    """
    if not HAVE_DEEPFACE or len(person_boxes) == 0:
        return "neutral", "Emotion defaulted to neutral (no DeepFace or no child)."

    # crop face from first person box
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_boxes[0]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return "neutral", "Face region empty; emotion set to neutral."

    try:
        result = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        # DeepFace usually returns a list
        if isinstance(result, list) and len(result) > 0:
            emo = result[0].get("dominant_emotion", "neutral")
        else:
            emo = result.get("dominant_emotion", "neutral")
        return emo, f"Detected child emotion: {emo}."
    except Exception as e:
        return "neutral", f"Emotion analysis failed; defaulting to neutral. ({e})"


def _get_dangerous_objects(dets: List[Dict], person_boxes: List, diag: float, use_llm: bool = True) -> Set[str]:
    """
    Get set of dangerous object classes using hybrid approach:
    1. Check obvious dangerous classes (instant)
    2. Use LLM to classify ambiguous objects near children (cached, fast)
    
    Args:
        dets: List of detections
        person_boxes: List of person bounding boxes
        diag: Diagonal of frame (for distance calculation)
        use_llm: Whether to use LLM for ambiguous objects (default: True)
    
    Returns:
        Set of dangerous object class names
    """
    global _llm_dangerous_cache
    
    # Get all unique object classes (excluding persons)
    object_classes = {d["class"] for d in dets if d["class"] != "person"}
    
    # Start with obvious dangerous objects (always check these)
    dangerous = {cls for cls in object_classes if cls.lower() in OBVIOUS_DANGEROUS_CLASSES}
    
    # For LLM classification, only check objects near children (to save time)
    # If no children, skip LLM check
    if use_llm and person_boxes:
        # Find objects near children
        ambiguous_near_children = set()
        for d in dets:
            cls = d["class"]
            if cls != "person" and cls.lower() not in OBVIOUS_DANGEROUS_CLASSES:
                # Check if object is near any child
                obj_center = _center(d["box"])
                for pb in person_boxes:
                    person_center = _center(pb)
                    dist = math.hypot(obj_center[0] - person_center[0], 
                                     obj_center[1] - person_center[1])
                    if dist < 0.3 * diag:  # Near threshold
                        ambiguous_near_children.add(cls)
                        break
        
        if ambiguous_near_children:
            try:
                from scene_explainer import classify_dangerous_objects
                llm_dangerous = classify_dangerous_objects(list(ambiguous_near_children), timeout=2.5)
                dangerous.update(llm_dangerous)
                # Update cache
                _llm_dangerous_cache.update(llm_dangerous)
            except Exception as e:
                print(f"[WARN] LLM danger classification failed: {e}")
                # Fall back to obvious only
    
    return dangerous


def classify_classroom_state(frame, dets: List[Dict], use_llm_danger_detection: bool = True):
    """
    Classify classroom state with hybrid dangerous object detection.
    
    Args:
        frame: BGR image frame
        dets: List of detections
        use_llm_danger_detection: Whether to use LLM for ambiguous objects (default: True)
    """
    h, w = frame.shape[:2]
    diag = math.hypot(w, h)

    person_boxes = []
    danger_boxes = []
    other_boxes = []

    # Get dangerous objects using hybrid approach
    # First pass: collect person boxes and categorize all detections
    for d in dets:
        cls = d["class"]
        box = d["box"]
        if cls == "person":
            person_boxes.append(box)
        else:
            # Will categorize after getting dangerous classes
            pass
    
    # Get dangerous objects (only check ambiguous objects near children with LLM)
    dangerous_classes = _get_dangerous_objects(dets, person_boxes, diag, use_llm=use_llm_danger_detection)

    # Second pass: categorize non-person objects
    for d in dets:
        cls = d["class"]
        box = d["box"]
        if cls != "person":  # Skip persons (already added)
            if cls in dangerous_classes:
                danger_boxes.append(box)
            else:
                other_boxes.append(box)

    num_persons = len(person_boxes)
    num_danger = len(danger_boxes)

    # Get main child's emotion (if any)
    emotion, emo_info = _get_child_emotion(frame, person_boxes)

    # ------------------------------------------------
    # 0) Empty classroom
    # ------------------------------------------------
    if num_persons == 0:
        reasons = "No person detected in the scene. " + emo_info
        return 3, "empty classroom", reasons

    # Helper: check distance between dangerous object and child
    def _dangerous_near_child():
        cnt = 0
        for db in danger_boxes:
            dcx, dcy = _center(db)
            for pb in person_boxes:
                pcx, pcy = _center(pb)
                dist = math.hypot(dcx - pcx, dcy - pcy)
                if dist < 0.25 * diag:  # 'near' threshold
                    cnt += 1
                    break
        return cnt

    # ------------------------------------------------
    # 1) Lone child logic
    # ------------------------------------------------
    if num_persons == 1:
        child_box = person_boxes[0]
        child_cx, child_cy = _center(child_box)

        dangerous_near = _dangerous_near_child()

        if dangerous_near > 0:
            reasons = (
                f"Single child detected with {dangerous_near} dangerous object(s) nearby. "
                f"{emo_info}"
            )
            # If child also looks scared/sad/angry, emphasize
            if emotion in NEGATIVE_EMOTIONS:
                reasons += " Child’s emotion appears negative; situation is more concerning."
            return 2, "dangerous object near child", reasons

        # No dangerous object, but lone child
        reasons = "Only one child detected in the classroom. " + emo_info
        if emotion in NEGATIVE_EMOTIONS:
            reasons += " Child appears emotionally upset or scared; needs care."
        elif emotion in POSITIVE_EMOTIONS:
            reasons += " Child appears in a positive mood but is still alone; teacher may check in."
        else:
            reasons += " Child appears neutral; still may need teacher attention."
        return 1, "lone child may need attention", reasons

    # ------------------------------------------------
    # 2) Multiple children present
    # ------------------------------------------------
    dangerous_near_child = _dangerous_near_child()
    if dangerous_near_child > 0:
        reasons = (
            f"Multiple children present; detected {dangerous_near_child} dangerous object(s) "
            "close to at least one child. " + emo_info
        )
        if emotion in NEGATIVE_EMOTIONS:
            reasons += " Main child’s emotion seems negative; situation may be stressful."
        return 2, "dangerous object near child", reasons

    # Check for isolation: very large min distance between any two students
    centers = [_center(b) for b in person_boxes]
    min_dist = diag
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = math.hypot(centers[i][0] - centers[j][0],
                           centers[i][1] - centers[j][1])
            if d < min_dist:
                min_dist = d

    lonely_child_flag = False
    if min_dist > 0.45 * diag:
        lonely_child_flag = True

    if lonely_child_flag:
        reasons = (
            "Multiple students detected, but one appears physically isolated from others. "
            + emo_info
        )
        if emotion in NEGATIVE_EMOTIONS:
            reasons += " Emotion suggests the isolated child may be unhappy or anxious."
        return 1, "lone child may need attention", reasons

    # ------------------------------------------------
    # 3) Default: normal class
    # ------------------------------------------------
    reasons = (
        "Multiple students present, no dangerous objects near children, and no obvious isolation. "
        + emo_info
    )
    return 0, "normal class activity", reasons
