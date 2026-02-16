"""
Vision Node - Distributed Vision-Control System (Phase 1)
---------------------------------------------------------
1. Captures frames from the webcam.
2. Detects and locks onto a target identity (using ArcFace).
3. Calculates position error (Is the face Left? Right? Centered?).
4. Publishes movement commands to the MQTT broker.

"""

from __future__ import annotations
import time
import json
import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import paho.mqtt.client as mqtt
import mediapipe as mp
from src.haar_5pt import align_face_5pt

# -------------------------
# CONFIGURATION
# -------------------------
# MQTT Settings
BROKER_ADDRESS = "157.173.101.159"
PORT = 1883
TEAM_ID = "y3c_ajhm"
TOPIC_MOVEMENT = f"vision/{TEAM_ID}/movement"

# Vision Settings
DEADZONE_THRESHOLD = 50  # Pixels from center to consider "CENTERED"
MIN_CONFIDENCE_LOCK = 0.65  # ArcFace similarity required to lock
MAX_LOST_FRAMES = 20  # Frames to keep lock without detection


# -------------------------
# Data Classes
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Helpers
# -------------------------
def calculate_iou(box_a: FaceDet, box_b: FaceDet) -> float:
    xA = max(box_a.x1, box_b.x1)
    yA = max(box_a.y1, box_b.y1)
    xB = min(box_a.x2, box_b.x2)
    yB = min(box_a.y2, box_b.y2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
    boxBArea = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def _clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    return x1, y1, x2, y2


def _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15):
    x_min, x_max = np.min(kps[:, 0]), np.max(kps[:, 0])
    y_min, y_max = np.min(kps[:, 1]), np.max(kps[:, 1])
    w, h = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
    return np.array([x_min - pad_x * w, y_min - pad_y_top * h,
                     x_max + pad_x * w, y_max + pad_y_bot * h], dtype=np.float32)


def _kps_span_ok(kps, min_eye_dist):
    le, re, no, lm, rm = kps
    if np.linalg.norm(re - le) < min_eye_dist: return False
    return lm[1] > no[1] and rm[1] > no[1]


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists(): return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}


# -------------------------
# Core Classes (Embedder, Detector, Matcher)
# -------------------------
class ArcFaceEmbedderONNX:
    def __init__(self, model_path="models/embedder_arcface.onnx"):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, img):
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.reshape(-1)
        return v / (np.linalg.norm(v) + 1e-12)


class HaarFaceMesh5pt:
    def __init__(self, min_size=(70, 70)):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.min_size = min_size
        self.idxs = [33, 263, 1, 61, 291]  # LE, RE, Nose, LM, RM

    def detect(self, frame, max_faces=5) -> List[FaceDet]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=self.min_size)
        if len(faces) == 0: return []

        # Sort by area (largest first)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[:max_faces]
        out = []

        for (x, y, w, h) in faces:
            # Crop ROI with padding
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.shape[0] < 20 or roi.shape[1] < 20: continue

            # Get landmarks
            res = self.mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks: continue

            lm = res.multi_face_landmarks[0].landmark
            kps = np.array([[lm[i].x * roi.shape[1] + rx1, lm[i].y * roi.shape[0] + ry1] for i in self.idxs],
                           dtype=np.float32)

            if not _kps_span_ok(kps, max(10.0, 0.18 * w)): continue

            bb = _bbox_from_5pt(kps)
            bx1, by1, bx2, by2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            out.append(FaceDet(bx1, by1, bx2, by2, 1.0, kps))

        return out


class FaceDBMatcher:
    def __init__(self, db):
        self._names = sorted(db.keys())
        self._mat = np.stack([db[n] for n in self._names]) if self._names else None

    def match(self, emb):
        if self._mat is None: return MatchResult(None, 1.0, 0.0, False)
        sims = (self._mat @ emb)
        idx = np.argmax(sims)
        sim = float(sims[idx])
        return MatchResult(self._names[idx] if sim > 0.6 else None, 1 - sim, sim, sim > 0.6)


# -------------------------
# MQTT & Control Logic
# -------------------------
class MqttManager:
    def __init__(self):
        self.client = mqtt.Client(client_id=f"vision_{TEAM_ID}_{int(time.time())}")
        self.client.on_connect = self.on_connect
        self.connected = False
        self.last_publish_time = 0

        print(f"📡 Connecting to MQTT Broker: {BROKER_ADDRESS}...")
        try:
            self.client.connect(BROKER_ADDRESS, PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"❌ MQTT Connection Error: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✅ MQTT Connected! Publishing to: {TOPIC_MOVEMENT}")
            self.connected = True
        else:
            print(f"❌ MQTT Connection Failed (Code: {rc})")

    def publish_movement(self, status: str, confidence: float):
        if not self.connected: return

        if time.time() - self.last_publish_time < 0.1: return

        payload = {
            "status": status,
            "confidence": float(round(confidence, 2)),
            "timestamp": int(time.time())
        }

        try:
            self.client.publish(TOPIC_MOVEMENT, json.dumps(payload))
            self.last_publish_time = time.time()
            # Print localized status for debugging
            print(f"📤 SENT: {status} ({confidence:.2f})")
        except Exception as e:
            print(f"⚠️ Publish Failed: {e}")


class FaceLockAndControl:
    def __init__(self, target_id):
        self.target_id = target_id
        self.locked = False
        self.locked_face = None
        self.lost_frames = 0

    def process(self, faces, frame, embedder, matcher, mqtt_mgr: MqttManager):
        h, w = frame.shape[:2]
        center_x = w // 2

        # 1. Update Lock State
        current_face = None

        if self.locked:
            # Try to find the same face based on IoU
            best_iou = 0
            best_match = None
            for f in faces:
                iou = calculate_iou(self.locked_face, f)
                if iou > best_iou:
                    best_iou = iou
                    best_match = f

            if best_match and best_iou > 0.3:
                self.locked_face = best_match
                self.lost_frames = 0
                current_face = best_match
            else:
                self.lost_frames += 1
                if self.lost_frames > MAX_LOST_FRAMES:
                    self.locked = False
                    self.locked_face = None
                    print(f"🔓 Lock Lost on {self.target_id}")
        else:
            # Search for target
            for f in faces:
                aligned, _ = align_face_5pt(frame, f.kps, (112, 112))
                res = matcher.match(embedder.embed(aligned))
                if res.accepted and res.name == self.target_id and res.similarity >= MIN_CONFIDENCE_LOCK:
                    self.locked = True
                    self.locked_face = f
                    self.lost_frames = 0
                    current_face = f
                    print(f"🔒 Lock Acquired: {self.target_id}")
                    break

        # 2. Determine Movement Command
        if current_face:
            face_cx = (current_face.x1 + current_face.x2) / 2
            diff = face_cx - center_x

            if abs(diff) < DEADZONE_THRESHOLD:
                status = "CENTERED"
            elif diff < 0:
                status = "MOVE_LEFT"  # Face is on the left
            else:
                status = "MOVE_RIGHT"  # Face is on the right

            # Publish with high confidence since we are locked
            mqtt_mgr.publish_movement(status, 1.0)

            # Draw UI
            color = (0, 255, 0) if status == "CENTERED" else (0, 165, 255)
            cv2.rectangle(frame, (current_face.x1, current_face.y1), (current_face.x2, current_face.y2), color, 3)
            cv2.putText(frame, f"{status}", (current_face.x1, current_face.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # No target face found
            mqtt_mgr.publish_movement("NO_FACE", 0.0)
            cv2.putText(frame, "SEARCHING...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# -------------------------
# Main
# -------------------------
def main():
    # 1. Load Database
    db_path = Path("data/db/face_db.npz")
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}. Please run src.enroll first.")
        return

    db = load_db_npz(db_path)
    if not db:
        print("❌ Database is empty.")
        return

    # 2. Select Target
    print("\nAvailable Identities:")
    names = list(db.keys())
    for i, name in enumerate(names):
        print(f"{i + 1}. {name}")

    idx = int(input("Select target ID number: ")) - 1
    target = names[idx]

    print(f"\n🚀 Starting Vision Node for Target: {target}")
    print(f"📡 MQTT Topic: {TOPIC_MOVEMENT}")
    print("Press 'q' to quit.\n")

    # 3. Initialize Components
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(db)
    mqtt_mgr = MqttManager()
    logic = FaceLockAndControl(target)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        faces = det.detect(frame)
        logic.process(faces, frame, embedder, matcher, mqtt_mgr)

        # Draw Center Line and Deadzone
        h, w = frame.shape[:2]

        cv2.imshow("Vision Node", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
