# scene_explainer.py (v3 - Qwen3-VL, improved prompting, temporal memory, smoothing)
"""
Upgraded scene explainer (v3) for Qwen3-VL vision narration.

Features:
- Improved structured prompt optimized for blind assistance
- Temporal memory to avoid repeated narration
- Multi-frame smoothing for stable motion/distance estimates
- Grouping, saliency ranking, hazard-first prompts
- Robust Ollama (Qwen3-VL) API integration with image + prompt
- Safe fallback narrator
"""

import os
import time
import math
import base64
import io
import re
import requests
from typing import List, Dict, Optional

# Model / endpoint configuration
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-vl:4b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT = OLLAMA_URL.rstrip("/") + "/api/chat"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 7.0))

# Small motion tracker & smoothing buffers (in-memory)
_TRACKER = {"history": [], "max_history": 6, "match_radius": 80}
_SMOOTH = {"rel_h": {}, "speed": {}, "rel_h_change": {}}

# Temporal narration memory (de-dup + rate-limit)
_LAST_NARRATION = {"text": "", "timestamp": 0.0, "min_gap": 2.5}

# Cache for LLM danger classification (class_name -> is_dangerous)
_DANGER_CACHE = {}


# ------------------------
# Utility helpers
# ------------------------
def _median_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _rel_box_height(box, H):
    x1, y1, x2, y2 = box
    return max(1, (y2 - y1)) / float(max(1, H))


def _horizontal_label(cx, W):
    if cx < W * 0.33:
        return "left"
    if cx > W * 0.66:
        return "right"
    return "center"


def _distance_label(rh):
    if rh > 0.42:
        return "very close"
    if rh > 0.20:
        return "near"
    if rh > 0.08:
        return "mid-distance"
    return "far"


def _speed_label(v):
    if v > 400:
        return "very fast"
    if v > 200:
        return "fast"
    if v > 60:
        return "moving"
    return "slow"


def _direction(dx, dy):
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    return "toward you" if dy < 0 else "away"


def _object_simple_id(d):
    """
    Lightweight pseudo-stable id for smoothing keyed by class+grid position.
    Not guaranteed stable across long occlusions but good enough for smoothing.
    """
    cx, cy = _median_center(d["box"])
    return f"{d['class']}_{int(cx/32)}_{int(cy/32)}"


def _smooth_value(dict_key: str, obj_id: str, new_val: float, alpha: float = 0.35):
    buf = _SMOOTH.setdefault(dict_key, {})
    if obj_id not in buf:
        buf[obj_id] = new_val
        return new_val
    prev = buf[obj_id]
    sm = prev * (1 - alpha) + new_val * alpha
    buf[obj_id] = sm
    return sm


# ------------------------
# Motion tracker (nearest neighbor between recent frames)
# ------------------------
def _update_tracker(dets: List[Dict], W: int, H: int):
    ts = time.time()
    curr = []
    for d in dets:
        cx, cy = _median_center(d["box"])
        rh = _rel_box_height(d["box"], H)
        curr.append({"class": d["class"], "center": (cx, cy), "rel_h": rh})

    hist = _TRACKER["history"]
    hist.append((ts, curr))
    if len(hist) > _TRACKER["max_history"]:
        hist.pop(0)

    motion = {}
    if len(hist) < 2:
        for i in range(len(curr)):
            motion[i] = {"speed": 0.0, "speed_label": "still", "direction": "stationary", "rel_h_change": 0.0}
        return motion

    t_old, prev = hist[-2]
    t_new, now = hist[-1]
    dt = max(1e-3, t_new - t_old)

    for i, c in enumerate(now):
        cx, cy = c["center"]
        best = None
        best_dist = float("inf")
        for p in prev:
            px, py = p["center"]
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best = p
                best_dist = dist

        if best is None or best_dist > _TRACKER["match_radius"]:
            motion[i] = {"speed": 0.0, "speed_label": "still", "direction": "stationary", "rel_h_change": 0.0}
            continue

        px, py = best["center"]
        dx, dy = cx - px, cy - py
        vx, vy = dx / dt, dy / dt
        s = math.hypot(vx, vy)
        motion[i] = {
            "speed": s,
            "speed_label": _speed_label(s),
            "direction": _direction(dx, dy),
            "rel_h_change": c["rel_h"] - best["rel_h"]
        }

    return motion


# ------------------------
# Observations builder (with smoothing)
# ------------------------
def detections_to_observations(dets: List[Dict], W: int, H: int):
    motion = _update_tracker(dets, W, H)
    observations = []
    for i, d in enumerate(dets):
        cx, cy = _median_center(d["box"])
        raw_rh = _rel_box_height(d["box"], H)
        obj_id = _object_simple_id(d)

        # smooth relative height and motion values
        rh = _smooth_value("rel_h", obj_id, raw_rh)
        mot = motion.get(i, {"speed": 0.0, "speed_label": "still", "direction": "stationary", "rel_h_change": 0.0})
        mot["speed"] = _smooth_value("speed", obj_id, mot.get("speed", 0.0))
        mot["rel_h_change"] = _smooth_value("rel_h_change", obj_id, mot.get("rel_h_change", 0.0))

        observations.append({
            "class": d.get("class", "object"),
            "conf": float(d.get("conf", 0.0)),
            "center": (cx, cy),
            "horiz": _horizontal_label(cx, W),
            "rel_h": rh,
            "distance": _distance_label(rh),
            "motion": mot
        })
    return observations


# ------------------------
# Grouping and saliency
# ------------------------
def group_observations(observations: List[Dict]):
    groups = {}
    for o in observations:
        key = (o["class"], o["horiz"], o["distance"])
        groups.setdefault(key, []).append(o)
    return groups


def importance_score(o: Dict):
    w_dist = {"very close": 4, "near": 3, "mid-distance": 2, "far": 1}
    w_speed = {"very fast": 4, "fast": 3, "moving": 2, "slow": 1}
    # larger rel_h implies closer/important
    size_score = 1.0 / (o.get("rel_h", 0.001) + 1e-6)
    return w_dist.get(o["distance"], 1) * 3 + w_speed.get(o["motion"].get("speed_label", "slow"), 1) * 2 + size_score


# ------------------------
# Danger heuristics
# ------------------------
def analyze_dangers(observations: List[Dict]) -> List[str]:
    msgs = []
    for o in observations:
        cls = o["class"].lower()
        dist = o["distance"]
        spd = o["motion"]["speed_label"]
        rel = o["motion"].get("rel_h_change", 0.0)
        if cls in ("car", "truck", "bus", "motorbike", "motorcycle", "bicycle"):
            if dist in ("very close", "near") and spd in ("fast", "very fast"):
                msgs.append(f"vehicle approaching {o['horiz']} — possible collision")
            elif dist in ("very close", "near"):
                msgs.append(f"{o['class']} {o['distance']} on the {o['horiz']}")
        if cls == "person":
            if rel > 0.03 and spd in ("fast", "very fast", "moving"):
                msgs.append(f"person approaching quickly on the {o['horiz']}")
            elif dist == "very close":
                msgs.append(f"person very close on the {o['horiz']}")
    return msgs


# ------------------------
# Build enhanced Qwen prompt
# ------------------------
SYSTEM_PROMPT = """
You are a concise, calm, and highly reliable visual narrator for blind or low-vision users.
Rules:
1) ALWAYS mention hazards first (if any).
2) Use simple language, short sentences (1-2 sentences total).
3) Use spatial words (left, right, center, directly ahead) and distance terms (very close, near, mid-distance, far).
4) Use movement words (approaching, moving away, moving left/right).
5) DO NOT guess identity, names, age, gender, or emotions. Do not hallucinate.
6) Do not mention image metadata or camera details.
7) If cinematic tone is requested, add at most one short evocative sentence after the safety-first sentence.
Return plain text only (no JSON).
""".strip()


def build_qwen_prompt(observations: List[Dict], hazards: List[str], cinematic: bool, scene_caption: Optional[str] = None):
    # Ordered (by saliency) observations text
    obs_sorted = sorted(observations, key=importance_score, reverse=True)
    obs_lines = []
    for o in obs_sorted[:8]:  # top 8 to keep prompt compact
        obs_lines.append(f"- {o['class']} ({int(o['conf']*100)}%) — {o['distance']} on the {o['horiz']}; motion: {o['motion']['speed_label']} ({o['motion']['direction']})")

    hazard_text = "No immediate hazards detected." if not hazards else "Potential hazards:\n" + "\n".join(f"- {h}" for h in hazards)
    caption_text = f"\nScene caption (optional): {scene_caption}\n" if scene_caption else "\n"

    user_prompt = f"""
{hazard_text}

Detected (top priority first):
{chr(10).join(obs_lines) if obs_lines else 'none'}
{caption_text}

Please produce a concise 1-2 sentence narration for a blind user. Mention hazards first if present.
Use the required spatial and distance language, be factual and avoid speculation.
""".strip()

    if cinematic:
        user_prompt += "\nAdd a subtle cinematic tone in a second short sentence only if it does not reduce clarity."

    return SYSTEM_PROMPT, user_prompt


# ------------------------
# Qwen (Ollama) call with image + prompt override
# ------------------------
def _call_qwen_with_image(pil_image, system_prompt: str, user_prompt: str, model: str = QWEN_MODEL, timeout: float = OLLAMA_TIMEOUT) -> Optional[str]:
    # encode PIL image as JPEG base64
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        img_b64 = None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    if img_b64:
        payload["images"] = [img_b64]

    try:
        r = requests.post(OLLAMA_CHAT, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # common Ollama responses: 'message' with 'content', 'response', 'output', 'choices'...
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "response" in data and isinstance(data["response"], str):
                return data["response"].strip()
            if "output" in data and isinstance(data["output"], str):
                return data["output"].strip()
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                # try nested structure
                c = data["choices"][0]
                if isinstance(c, dict) and "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
                    return c["message"]["content"].strip()
        # fallback: find first non-empty string value
        for v in (data.values() if isinstance(data, dict) else []):
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        return None
    return None


# ------------------------
# LLM-based danger classification (fast with caching)
# ------------------------
def classify_dangerous_objects(object_classes: List[str], timeout: float = 3.0) -> set:
    """
    Classify which objects are dangerous in a classroom context using LLM.
    Uses caching to avoid repeated LLM calls for the same object class.
    
    Args:
        object_classes: List of object class names to check
        timeout: Timeout for LLM call (shorter than narration for speed)
    
    Returns:
        Set of object class names that are dangerous
    """
    dangerous = set()
    to_check = []
    
    # Check cache first
    for cls in object_classes:
        cls_lower = cls.lower()
        if cls_lower in _DANGER_CACHE:
            if _DANGER_CACHE[cls_lower]:
                dangerous.add(cls)
        else:
            to_check.append(cls)
    
    # If all cached, return immediately
    if not to_check:
        return dangerous
    
    # Batch check remaining objects with LLM
    if to_check:
        # Create a simple text-only prompt (no image needed for class names)
        prompt = f"""You are a safety expert for elementary school classrooms.

Given these object types detected in a classroom, classify which ones could be DANGEROUS for children:
{', '.join(to_check)}

Consider:
- Sharp objects (knives, scissors, etc.)
- Small objects that could be choking hazards
- Objects that could cause injury if misused
- Objects that are inappropriate for children

Respond with ONLY a comma-separated list of dangerous object names (use exact names from the list above).
If none are dangerous, respond with "none".

Example response: "knife, scissors, small toy"
"""
        
        try:
            payload = {
                "model": QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a safety expert. Respond with only a comma-separated list of dangerous object names, or 'none'."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            r = requests.post(OLLAMA_CHAT, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            
            # Extract response
            response = ""
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                    response = data["message"]["content"].strip()
                elif "response" in data and isinstance(data["response"], str):
                    response = data["response"].strip()
                elif "output" in data and isinstance(data["output"], str):
                    response = data["output"].strip()
            
            # Parse response
            if response and response.lower() != "none":
                # Extract object names from response
                response_lower = response.lower()
                for cls in to_check:
                    cls_lower = cls.lower()
                    # Check if class name appears in response
                    if cls_lower in response_lower:
                        dangerous.add(cls)
                        _DANGER_CACHE[cls_lower] = True
                    else:
                        _DANGER_CACHE[cls_lower] = False
            else:
                # No dangerous objects
                for cls in to_check:
                    _DANGER_CACHE[cls.lower()] = False
                    
        except Exception as e:
            # On error, default to safe (don't mark as dangerous)
            print(f"[WARN] LLM danger classification failed: {e}")
            for cls in to_check:
                _DANGER_CACHE[cls.lower()] = False
    
    return dangerous


# ------------------------
# Fallback narrator
# ------------------------
def fallback_narrate(observations: List[Dict], cinematic: bool) -> str:
    if not observations:
        return "No prominent objects nearby."
    dangers = analyze_dangers(observations)
    if dangers:
        return "Warning: " + dangers[0] + "."
    # group a few important items
    obs_sorted = sorted(observations, key=importance_score, reverse=True)
    phrases = []
    for o in obs_sorted[:3]:
        phrases.append(f"{o['class']} {o['distance']} on the {o['horiz']}")
    text = ". ".join(phrases) + "."
    if cinematic:
        text += " Scene is calm." if all(o["motion"]["speed_label"] == "slow" for o in observations) else " Scene shows movement."
    return text


# ------------------------
# Temporal narration memory: suppress repeats & rate-limit
# ------------------------
def should_output_narration(text: str) -> bool:
    global _LAST_NARRATION
    t = time.time()
    if not text or not text.strip():
        return False
    s = text.strip()
    # Avoid repeating same text exactly
    if s.lower() == _LAST_NARRATION["text"].lower():
        return False
    # Enforce minimal gap
    if t - _LAST_NARRATION["timestamp"] < _LAST_NARRATION["min_gap"]:
        return False
    # Passed checks: update memory and allow output
    _LAST_NARRATION["text"] = s
    _LAST_NARRATION["timestamp"] = t
    return True


# ------------------------
# Main exported function
# ------------------------
def explain_scene(
    pil_image,
    detections: List[Dict],
    frame_w: int,
    frame_h: int,
    scene_caption: Optional[str] = None,
    cinematic: bool = False,
    detect_actions: bool = True,
    detect_emotions: bool = True,
    danger_check: bool = True,
    prompt_override: Optional[str] = None,
    bypass_suppression: bool = False
) -> str:
    """
    - pil_image: PIL.Image of the current frame (JPEG-compatible)
    - detections: list of {'class','conf','box':(x1,y1,x2,y2)}
    - frame_w, frame_h: width, height of frame
    - cinematic: if True, allow slight cinematic tone in second sentence
    - prompt_override: optional string to use as the user prompt (skips automatic prompt builder)
    - bypass_suppression: if True, always return AI response even if it's a repeat (for display purposes)
    Returns: narration string (possibly empty if suppressed by memory, unless bypass_suppression=True)
    """
    observations = detections_to_observations(detections, frame_w, frame_h)

    # quick local hazard check (urgent hazards are returned immediately)
    hazards = analyze_dangers(observations) if danger_check else []
    if hazards:
        urgent = "Warning: " + hazards[0] + "."
        # return urgent immediately but still enforce repetition suppression
        if bypass_suppression or should_output_narration(urgent):
            return urgent
        return ""  # suppressed by memory

    # Build prompt
    if prompt_override:
        system_prompt = SYSTEM_PROMPT
        user_prompt = prompt_override
    else:
        system_prompt, user_prompt = build_qwen_prompt(observations, hazards, cinematic, scene_caption)

    # Call Qwen3-VL with image + structured prompt
    resp = _call_qwen_with_image(pil_image, system_prompt, user_prompt, model=QWEN_MODEL, timeout=OLLAMA_TIMEOUT)

    if resp:
        # Keep only 1-2 sentences
        sentences = re.split(r'(?<=[.!?])\s+', resp.strip())
        top = " ".join(sentences[:2]).strip()
        if bypass_suppression:
            # Always return AI response, but still update memory to track it
            if top:
                _LAST_NARRATION["text"] = top.strip().lower()
                _LAST_NARRATION["timestamp"] = time.time()
            return top
        if should_output_narration(top):
            return top
        return ""

    # Fallback narrator
    fb = fallback_narrate(observations, cinematic)
    if bypass_suppression:
        # Always return fallback if AI failed
        if fb:
            _LAST_NARRATION["text"] = fb.strip().lower()
            _LAST_NARRATION["timestamp"] = time.time()
        return fb
    if should_output_narration(fb):
        return fb
    return ""
