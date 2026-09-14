import cv2
import torch
import torch.nn as nn
import json
import numpy as np
import mediapipe as mp   
from torchvision import models, transforms
from PIL import Image
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
with open("models/classes.json") as f:
    classes = json.load(f)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/sign_cnn.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Hand detector (to only classify when hands are visible)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# TTS
import pyttsx3
tts = pyttsx3.init()
tts.setProperty('rate', 150)

def speak(text):
    tts.say(text)
    tts.runAndWait()

# Main loop
cap = cv2.VideoCapture(0)
global_ts = [0]
last_prediction = ""
cooldown = 0

print("SignSpeak LIVE — show a sign to the camera! (Ctrl+C to stop)")
print(f"Recognizing {len(classes)} signs: {', '.join(classes[:10])}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check for hands
    ts = global_ts[0]
    global_ts[0] += 33
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, ts)
    has_hands = len(result.hand_landmarks) > 0

    if has_hands and cooldown <= 0:
        # Classify
        img = cv2.resize(frame, (224, 224))
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            conf, pred = torch.max(torch.softmax(logits, dim=1), 1)
            pred = pred.item()
            conf = conf.item()

        if conf > 0.05:
            label = classes[pred]
            if label != last_prediction:
                last_prediction = label
                cooldown = 30  # ignore for ~1 second
                print(f"\n✓ {label} ({conf*100:.0f}%)")
                speak(label)
    else:
        cooldown -= 1

    # Draw on frame
    if has_hands:
        cv2.putText(frame, f"Detected: {last_prediction} ({conf*100:.0f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Show your hands...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("SignSpeak", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()   