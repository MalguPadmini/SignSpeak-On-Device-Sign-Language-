import mediapipe as mp
import cv2
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = r"data\include\ProcessedData_vivit"
OUTPUT_DIR = r"data\landmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear old files
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.npy'):
        os.remove(os.path.join(OUTPUT_DIR, f))

global_ts = [0]

def extract_video_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        ts = global_ts[0]
        global_ts[0] += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, ts)

        landmarks = np.zeros((42, 3))
        if result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                for j, lm in enumerate(hand):
                    landmarks[i*21 + j] = [lm.x, lm.y, lm.z]

        # ONLY keep frames where at least one hand is detected
        if np.count_nonzero(landmarks) >= 21:
            frames.append(landmarks)
    cap.release()

    # Need at least 15 valid frames
    if len(frames) < 15:
        return None

    # Pad to 30 if needed
    while len(frames) < 30:
        frames.append(frames[-1])
    return np.array(frames[:30])

signs = os.listdir(DATA_DIR)
print(f"Processing {len(signs)} signs...")
saved = 0

for sign in sorted(signs):
    sign_path = os.path.join(DATA_DIR, sign)
    if not os.path.isdir(sign_path):
        continue
    videos = [f for f in os.listdir(sign_path) if f.endswith('.MOV')]
    count = 0
    for i, vid in enumerate(videos):
        landmarks = extract_video_landmarks(os.path.join(sign_path, vid))
        if landmarks is not None:
            np.save(os.path.join(OUTPUT_DIR, f"{sign}_{i:02d}.npy"), landmarks)
            count += 1
    saved += count
    print(f"{sign}: {count}/{len(videos)} valid")

print(f"\nDone! {saved} valid samples saved to {OUTPUT_DIR}")
detector.close()   