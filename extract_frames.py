import cv2
import os

DATA_DIR = r"data\include\ProcessedData_vivit"
OUTPUT_DIR = r"data\frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

signs = sorted(os.listdir(DATA_DIR))
print(f"Extracting frames from {len(signs)} signs...")
saved = 0

for sign in signs:
    sign_path = os.path.join(DATA_DIR, sign)
    if not os.path.isdir(sign_path):
        continue
    videos = [f for f in os.listdir(sign_path) if f.endswith('.MOV')]
    count = 0
    for i, vid in enumerate(videos):
        vid_path = os.path.join(sign_path, vid)
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            cap.release()
            continue
        # Jump to middle frame
        mid = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{sign}_{i:02d}.jpg"), frame)
            count += 1
    saved += count
    print(f"{sign}: {count}/{len(videos)}")

print(f"\nDone! {saved} frames saved to {OUTPUT_DIR}")   