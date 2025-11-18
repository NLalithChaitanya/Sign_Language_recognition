
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# --- 1. Define CNN Model ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 24)  # 24 letters (A-Y, no J & Z)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load("sign_mnist_model.pth", map_location=device))
model.eval()

# --- 3. Transformations ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 4. Letter Mapping ---
classes = ['A','B','C','D','E','F','G','H','I','K','L','M',
           'N','O','P','Q','R','S','T','U','V','W','X','Y']

# --- 5. Initialize Variables ---
cap = cv2.VideoCapture(0)
word = ""
current_letter = None
letter_buffer = []
buffer_size = 10  # Collect 10 consistent predictions
confidence_threshold = 0.75
last_add_time = 0
min_time_between_letters = 1.0  # seconds

def get_stable_prediction(buffer):
    """Return most common letter if it appears enough times"""
    if len(buffer) < buffer_size:
        return None
    
    # Count occurrences
    counts = {}
    for letter in buffer:
        counts[letter] = counts.get(letter, 0) + 1
    
    # Return letter if it appears in 70% of buffer
    for letter, count in counts.items():
        if count >= buffer_size * 0.7:
            return letter
    return None

print("=== Sign Language Recognition ===")
print("Controls:")
print("  SPACE - Add space to word")
print("  BACKSPACE - Delete last character")
print("  C - Clear word")
print("  Q - Quit")
print("================================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ROI - centered square
    roi_size = 300
    x = (w - roi_size) // 2
    y = (h - roi_size) // 2
    roi = frame[y:y+roi_size, x:x+roi_size]

    # --- Preprocess ROI ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    predicted_letter = None
    confidence = 0.0
    
    # Check for hand presence
    if cv2.countNonZero(thresh) > 800:
        try:
            img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                probs = F.softmax(output, dim=1)
                conf, predicted = torch.max(probs, 1)

            confidence = conf.item()
            if confidence > confidence_threshold:
                predicted_letter = classes[predicted.item()]
                letter_buffer.append(predicted_letter)
                
                # Keep buffer at fixed size
                if len(letter_buffer) > buffer_size:
                    letter_buffer.pop(0)
        except Exception as e:
            print(f"Prediction error: {e}")

    # Check for stable prediction
    stable_letter = get_stable_prediction(letter_buffer)
    current_time = time.time()
    
    if (stable_letter and 
        stable_letter != current_letter and 
        current_time - last_add_time >= min_time_between_letters):
        word += stable_letter
        current_letter = stable_letter
        last_add_time = current_time
        letter_buffer.clear()
        print(f"Added: {stable_letter} | Word: {word}")

    # --- Drawing UI ---
    # ROI rectangle
    color = (0, 255, 0) if predicted_letter else (128, 128, 128)
    cv2.rectangle(frame, (x, y), (x+roi_size, y+roi_size), color, 3)

    # Prediction text
    if predicted_letter:
        cv2.putText(frame, f'Detected: {predicted_letter}', 
                    (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        
        # Confidence bar
        bar_width = int(roi_size * confidence)
        cv2.rectangle(frame, (x, y-10), (x+bar_width, y-5), (0, 255, 0), -1)

    # Buffer status
    buffer_fill = len(letter_buffer)
    cv2.putText(frame, f'Buffer: {buffer_fill}/{buffer_size}', 
                (x, y+roi_size+25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (200, 200, 200), 2)

    # Word display
    cv2.rectangle(frame, (10, h-80), (w-10, h-10), (0, 0, 0), -1)
    cv2.putText(frame, f'Word: {word}', 
                (20, h-35), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, (0, 255, 255), 3)

    # Instructions
    cv2.putText(frame, 'SPACE: space | BACKSPACE: delete | C: clear | Q: quit', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1)

    cv2.imshow("Sign Language Recognition", frame)

    # --- Key Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Add space
        if word and word[-1] != ' ':
            word += ' '
            current_letter = None
            letter_buffer.clear()
            print(f"Added space | Word: {word}")
    elif key == 8:  # Backspace
        if word:
            word = word[:-1]
            print(f"Deleted | Word: {word}")
    elif key == ord('c'):  # Clear
        print(f"Final word: {word}")
        word = ""
        current_letter = None
        letter_buffer.clear()
        last_add_time = 0

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal word: {word}")