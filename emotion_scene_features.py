# emotion_scene_features.py
import cv2
import numpy as np

try:
    from deepface import DeepFace
except:
    DeepFace = None
    print("[WARN] DeepFace missing – emotion analysis disabled, defaulting to neutral.")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def extract_emotion_vector(frame, person_boxes):
    """Returns a one-hot emotion vector."""
    emo = {f"emotion_{e}": 0 for e in EMOTIONS}
    dominant = "neutral"

    if DeepFace is None or len(person_boxes) == 0:
        emo["emotion_neutral"] = 1
        return emo

    x1, y1, x2, y2 = person_boxes[0]
    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        emo["emotion_neutral"] = 1
        return emo

    try:
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        dominant = result[0]["dominant_emotion"]
    except:
        dominant = "neutral"

    if dominant in EMOTIONS:
        emo[f"emotion_{dominant}"] = 1
    else:
        emo["emotion_neutral"] = 1

    return emo


def extract_scene_features(dets, shape):
    """Extract simple scene understanding features."""
    h, w = shape[:2]
    img_area = float(h * w)

    person_boxes = []
    other_boxes = []
    total_area = 0

    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cls = d["class"]
        area = (x2 - x1) * (y2 - y1)
        total_area += max(0, area)

        if cls == "person":
            person_boxes.append((x1, y1, x2, y2))
        else:
            other_boxes.append((x1, y1, x2, y2))

    num_persons = len(person_boxes)
    num_objects = len(other_boxes)
    occupancy = total_area / (img_area + 1e-6)

    min_person_dist = w
    if len(person_boxes) > 1:
        centers = []
        for x1, y1, x2, y2 in person_boxes:
            centers.append(((x1+x2)/2, (y1+y2)/2))

        dists = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                d = ((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)**0.5
                dists.append(d)

        if dists:
            min_person_dist = min(dists)

    return {
        "num_persons": num_persons,
        "num_objects": num_objects,
        "occupancy_ratio": occupancy,
        "min_person_dist": min_person_dist,
        "has_multiple_people": 1 if num_persons > 1 else 0
    }, person_boxes


def extract_full_feature_vector(frame, dets):
    """Full scene + emotion stack used for ML."""
    scene, persons = extract_scene_features(dets, frame.shape)
    emotion = extract_emotion_vector(frame, persons)
    full = {**scene, **emotion}
    return full
