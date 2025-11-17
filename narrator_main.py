# narrator_main.py
"""
Classroom Safety Narrator (Live Video or Uploaded Video)
- Supports webcam (default) or a video file path.
- Scene understanding + emotion + rule-based safety logic.
"""

import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from classroom_rules import classify_classroom_state
from scene_explainer import explain_scene   # your existing LLM narrator


MODEL_NAME = "yolov8n.pt"
FRAME_SKIP = 2   # Skip frames to increase speed


class Narrator:
    def __init__(self):
        print("[INFO] Loading YOLO model:", MODEL_NAME)
        self.model = YOLO(MODEL_NAME)

    def detect(self, frame):
        results = self.model(frame)
        r = results[0]

        dets = []
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = self.model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, xyxy)

            dets.append({
                "class": cls_name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

        return dets

    def narrate(self, frame):
        dets = self.detect(frame)

        # SCENE + EMOTION + RULE-BASED SAFETY
        state_id, state_text, reasons = classify_classroom_state(frame, dets)

        # Pass the situation to LLM for narration
        hint = (
            f"Classroom safety state: {state_text}. "
            f"Reason: {reasons}. Provide a calm explanation."
        )

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Always get AI narration output (bypass suppression for display purposes)
        narration = explain_scene(
            pil_img,
            dets,
            frame.shape[1],
            frame.shape[0],
            scene_caption=hint,
            cinematic=False,
            detect_actions=True,
            detect_emotions=False,
            danger_check=True,
            bypass_suppression=True  # Always return AI model output
        )

        return dets, narration, state_text


def run(video_source=None):
    narrator = Narrator()

    # If no argument → use webcam
    if video_source is None:
        print("[INFO] Starting LIVE WEBCAM mode.")
        cap = cv2.VideoCapture(2)
    else:
        print("[INFO] Opening video file:", video_source)
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open camera or video file.")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        dets, narration, state_text = narrator.narrate(frame)

        # Draw YOLO boxes
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            label = f"{d['class']} {d['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Compose overlay text
        header = f"[STATE: {state_text.upper()}] {narration}"
        if len(header) > 260:
            header = header[:260] + "..."

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, header, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2)

        cv2.imshow("Classroom Safety Narrator (Live/Video)", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])     # Provided video file
    else:
        run(None)            # Webcam mode
