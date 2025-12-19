# Sign Language Recognition Using CNNs

This project implements a **Convolutional Neural Network (CNN)**–based system for **static American Sign Language (ASL) alphabet recognition** using the **Sign Language MNIST** dataset. The model achieves **97.55% test accuracy** and supports training, evaluation, visualization, and real-time webcam inference.

---

## Project Overview

Sign language recognition plays an important role in improving accessibility for the hearing-impaired community.  
This project focuses on recognizing **static ASL hand gestures** (alphabets A–Y, excluding J and Z) using deep learning.

The system:
- Trains a CNN on grayscale hand images
- Evaluates performance using accuracy, classification reports, and confusion matrices
- Performs real-time prediction using a webcam with MediaPipe-based hand detection

---

## Dataset

- **Dataset:** Sign Language MNIST  
- **Classes:** 24 static ASL letters (A–Y, excluding J and Z)  
- **Image size:** 28 × 28 grayscale  
- **Training samples:** 27,455  
- **Test samples:** 7,172  

The dataset is provided in CSV format and converted to images locally.  
Dataset files and generated images are **not included in the repository**.

---

## Model Architecture

The CNN consists of:
- Two convolutional layers with Batch Normalization
- Max pooling layers for downsampling
- Fully connected layers with Dropout for regularization
- Softmax output for multi-class classification

Training configuration:
- **Loss function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch size:** 64  
- **Epochs:** 10  

---

## Results

- **Best Test Accuracy:** **97.55%**
- Stable convergence with minimal overfitting
- Near-perfect performance for most classes
- Minor confusion between visually similar signs (e.g., R and T)

Evaluation includes:
- Training vs Validation Loss
- Training vs Validation Accuracy
- Normalized Confusion Matrix
- Per-Class Accuracy Analysis

---

## Real-Time Inference

The project supports real-time ASL recognition using a webcam:
- MediaPipe Hands detects and crops the hand region
- The cropped image is preprocessed to match training conditions
- The CNN predicts the corresponding ASL letter
- Temporal smoothing improves prediction stability

**Note:**  
The model is trained on clean, centered dataset images. Performance may degrade on real-world webcam input due to lighting variations, background noise, and viewpoint differences (domain shift).

---

## Project Structure

Sign_Language_recognition/
│
├── main.py # Training and evaluation pipeline
├── sign_lang.ipynb # Experiments and visualizations
├── requirements.txt # Python dependencies
├── classes.json # Class labels
├── .gitignore # Ignored datasets, images, models
└── README.md # Project documentation

## Limitations

- The system supports **only static ASL signs** and does not handle dynamic gestures.
- Letters **J and Z** are excluded, as they require motion-based recognition.
- The model is trained on **clean, centered dataset images**, which leads to reduced accuracy on real-world webcam input due to **domain shift**.
- Performance is sensitive to **lighting conditions, hand orientation, background clutter, and camera quality**.
- The CNN relies on **pixel-level features**, making it vulnerable to noise and background variations.

---

## Future Work

- Extend the system to recognize **dynamic ASL gestures** using temporal models such as **LSTMs or Transformers**.
- Replace pixel-based inputs with **MediaPipe hand keypoints** to improve robustness against lighting and background changes.
- Apply **data augmentation** and fine-tune the model on real-world webcam data to reduce domain shift.
- Optimize the pipeline for real-time performance and deploy it as a **web or mobile application**.
- Explore **continuous sign sequence recognition** for full sign language translation.

Generated data such as images, datasets, and trained models are intentionally excluded from version control.

---

## Setup and Usage

### Clone the repository

```bash
git clone https://github.com/NLalithChaitanya/Sign_Language_recognition.git
cd Sign_Language_recognition

Create and activate environment

bash
conda create -n slr python=3.9
conda activate slr
pip install -r requirements.txt

Run real-time inference

bash
python main.py

Training, Testing, Experiments and Visualizations
All experiments, training, testing, and visualizations are available in:

sign_lang.ipynb
