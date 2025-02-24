# Facial Expression Recognition
This project implements a pre-trained facial expression recognition model inspired by the work of the ibug-group. It includes additional features for data handling and grayscale image processing.  

## Prerequisites
* [Numpy](https://www.numpy.org/): numpy>=1.16.0
* [PyTorch](https://pytorch.org/): torch>=1.1.0
* [OpenCV](https://opencv.org/): opencv-python>= 3.4.2
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script). See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).
* [ibug.face_alignment](https://github.com/hhj1897/face_alignment). See this repository for details: [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment)  

## How to Test
* To test on live video: `python emotion_recognition_test.py [-i webcam_index]`
* To test on a video file: `python emotion_recognition_test.py [-i input_file] [-o output_file]`

## Features
1. Face detection using RetinaFace or S3FD methods
2. Facial landmark detection using FAN (Face Alignment Network)
3. Emotion recognition using EmoNet
4. Real-time processing of video streams or files
5. Output of processed video with detected faces, landmarks, and emotions
6. Generation of CSV files with frame-by-frame emotion data

## Usage
* `--input` or `-i`: Specify the input video file or webcam index (default is 0 for the first webcam)
* `--output` or `-o`: Set the output video file path
* `--detection-method` or `-dm`: Choose the face detection method (RetinaFace or S3FD)
For a full list of available options, run:
`python emotion_recognition_test_gray.py --help`

## Output
1. `table1.csv`: Contains frame numbers and detected emotions
2. `table2.csv`: Contains frame numbers, detected emotions, arousal and valence

## Citation
This project is based on the work of Jie Shen found [here](https://github.com/hhj1897/emotion_recognition).  
Their project is also based on the official implementation of the paper "Estimation of continuous valence and arousal levels from faces in naturalistic conditions", Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, Nature Machine Intelligence, 2021, found [here](https://github.com/face-analysis/emonet).

# Analysis Script
## Features
* Calculates average arousal and valence for first and last thirds of the data
* Generates plots for arousal and valence over time
* Applies moving average smoothing
* Performs linear regression to identify trends

## Usage
  1. Run the script: `python analysis.py`
  2. Enter the input file name (without .csv extension)

## Outputs
* `analysis.csv`: Average arousal and valence values
* `arousal.png`: Plot of arousal over time
* `valence.png`: Plot of valence over time
* `smooth.png`: Combined plot with smoothed data and trend lines
All outputs are saved in the same directory as the input file.



