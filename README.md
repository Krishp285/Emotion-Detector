# Real-Time Emotion Detector
Detects facial emotions live using a CNN and OpenCV.

## Overview
- **Accuracy**: 55.74% on FER-2013 dataset
- **Tools**: Python 3.12, TensorFlow 2.19.0, OpenCV 4.11.0.86
- **Features**: Trains a CNN on FER-2013, detects emotions (Angry, Happy, etc.) via webcam

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and extract to `train/` and `test/`
3. Train: `py -3.12 emotion_detector.py`
4. Run live: `py -3.12 detect_emotions_live.py`

## Files
- `emotion_detector.py`: Trains the CNN
- `detect_emotions_live.py`: Live webcam detection
- `emotion_model.h5`: Pre-trained model (55.74% accuracy)
