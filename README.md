# SignSpeak

**Real-Time Sign Language Recognition Using Computer Vision and Deep Learning**

SignSpeak is a computer vision and deep learning project designed to recognize sign language gestures in real time through a webcam. The system processes hand movements and gestures using computer vision techniques and predicts the corresponding sign, making communication more accessible through automatic text and speech output.

## Overview

Sign language provides an essential means of communication for millions of people worldwide. However, communication between sign language users and people unfamiliar with sign language can often be challenging.

SignSpeak aims to bridge this communication gap by providing a real-time sign recognition system that can:

* Detect hands using computer vision.
* Extract meaningful hand landmarks.
* Analyze hand gestures and movements.
* Classify recognized signs using deep learning.
* Display the predicted sign in real time.
* Convert recognized signs into speech using text-to-speech.

## Key Features

* Real-time webcam-based sign recognition
* Hand detection and landmark extraction using MediaPipe
* Deep learning-based gesture classification
* CNN-based image classification
* LSTM-based temporal sequence classification
* Real-time prediction with confidence scores
* Text-to-speech output
* Modular training and inference pipeline

## System Architecture

```text
                 Webcam Input
                      |
                      v
             Hand Detection
                (MediaPipe)
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Hand Landmarks            Video Frames
          |                       |
          v                       v
    LSTM Classifier           CNN Model
          |                       |
          +-----------+-----------+
                      |
                      v
              Sign Prediction
                      |
              +-------+-------+
              |               |
              v               v
             Text          Speech
```

## Technologies Used

| Technology  | Purpose                                    |
| ----------- | ------------------------------------------ |
| Python      | Core programming language                  |
| OpenCV      | Video capture and image processing         |
| MediaPipe   | Hand detection and landmark extraction     |
| PyTorch     | Deep learning model development            |
| Torchvision | CNN architecture and image transformations |
| NumPy       | Numerical and dataset processing           |
| Pillow      | Image processing                           |
| pyttsx3     | Text-to-speech conversion                  |

## Models

### CNN

A ResNet18-based convolutional neural network is used for image-based sign classification.

The CNN processes individual video frames and learns visual features associated with different hand signs.

### LSTM

An LSTM-based classifier is used for temporal gesture recognition.

Instead of analyzing a single frame, the LSTM processes a sequence of hand landmark coordinates across multiple frames, allowing it to capture movement and temporal information.

## Project Structure

```text
SignSpeak/
│
├── data/
│   ├── frames/
│   └── landmarks/
│
├── models/
│
├── extract_frames.py
├── extract_landmarks.py
├── train_classifier.py
├── train_cnn.py
├── demo.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/MalguPadmini/SignSpeak-On-Device-Sign-Language-.git
cd SignSpeak
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start the real-time recognition system:

```bash
python demo.py
```

The application will access the webcam and begin processing the detected hand gestures.

Press `Q` to exit the application.

## Training Pipeline

The project follows a multi-stage data and training pipeline.

### 1. Extract Video Frames

```bash
python extract_frames.py
```

This extracts individual frames from the input sign language videos.

### 2. Extract Hand Landmarks

```bash
python extract_landmarks.py
```

MediaPipe is used to detect hands and extract their landmark coordinates from the video frames.

### 3. Train the LSTM Classifier

```bash
python train_classifier.py
```

The extracted landmark sequences are used to train the LSTM-based sign classifier.

### 4. Train the CNN

```bash
python train_cnn.py
```

The extracted image frames are used to train the ResNet18-based CNN classifier.

## Workflow

The complete workflow can be summarized as:

```text
Input Videos
     |
     v
Frame Extraction
     |
     +-------------------+
     |                   |
     v                   v
Image Dataset      Hand Landmark Dataset
     |                   |
     v                   v
ResNet18 CNN        LSTM Classifier
     |                   |
     +---------+---------+
               |
               v
       Real-Time Prediction
               |
               v
          Text Output
               |
               v
       Text-to-Speech
```

## Use Cases

SignSpeak can serve as a foundation for applications such as:

* Assistive communication systems
* Sign language learning tools
* Accessibility-focused applications
* Human-computer interaction
* Educational technology
* Real-time gesture-based interfaces

## Future Improvements

Potential improvements include:

* Expanding the sign language vocabulary
* Improving recognition accuracy under different lighting conditions
* Supporting continuous sentence-level recognition
* Improving multi-hand gesture recognition
* Adding a graphical user interface
* Deploying the model as a web or mobile application
* Incorporating a larger and more diverse dataset
* Improving robustness across different users and backgrounds

## Limitations

The performance of the system depends on factors such as:

* Dataset size and diversity
* Lighting conditions
* Camera quality
* Hand visibility
* Background complexity
* Similarity between different gestures

The current implementation is primarily intended as a research and development project and can be further improved for large-scale real-world deployment.

## License

This project is intended for educational and research purposes.by me.

## Author

**SignSpeak**

A computer vision and deep learning project focused on making sign language recognition more accessible through real-time AI-based gesture recognition.
