
import cv2
import torch
import numpy as np
import json
from collections import deque
from model import CNNModel3
import mediapipe as mp

# Load classes.json

with open("classes.json", "r") as f:
    classes = json.load(f)
idx_to_class = {i: cls for i, cls in enumerate(classes)}


# Load model

checkpoint = torch.load("best_signmnist.pth", map_location="cpu")
model = CNNModel3(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Setup MediaPipe Hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# Webcam setup

cap = cv2.VideoCapture(0)
cap.set(3, 700)
cap.set(4, 480)

print("Webcam ready... Press 'q' to quit.")


# Frame smoothing

SMOOTH_WINDOW = 5
pred_queue = deque(maxlen=SMOOTH_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get bounding box of hand
        h, w, c = frame.shape
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Add some padding
        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        roi = frame[y_min:y_max, x_min:x_max]

        # Preprocess for model
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = cv2.equalizeHist(img)
        img = img.astype("float32") / 255.0
        img = np.reshape(img, (1, 1, 28, 28))
        img_tensor = torch.from_numpy(img)

        # Predict
        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1)
            prob, pred_idx = torch.max(probs, 1)

        prob = float(prob.item())
        pred_idx = int(pred_idx)

        # Frame smoothing
        pred_queue.append(pred_idx)
        pred_idx_smooth = max(set(pred_queue), key=pred_queue.count)

        # Display prediction
        if prob < 0.4:
            text = "Sign not detected"
        else:
            predicted_letter = idx_to_class[pred_idx_smooth]
            text = f"{predicted_letter}: {prob*100:.2f}%"

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
