# OpenCV Function Explorer

OpenCV Function Explorer is a Python application that allows users to explore and apply various OpenCV functions on images and videos. The application supports live video processing, Video Processing, Image Display and YOLO-based object detection.

## Features

- Load and apply YOLO models for object detection.
- Select and apply different image filters.
- Process and display video files or live video feeds.
- GUI with Tkinter for user-friendly interaction.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/muhammadmustafaisb/Object-Detection-Using-Yolo
   cd Object-Detection-Using-Yolo
   ```

2. Create a virtual environment and activate it (optional but recommended):

   ```sh
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```sh
   pip install opencv-python numpy pillow
   ```

4. Run the application:

   ```sh
   python main.py
   ```
   
## Usage

1. **Download YOLO Files:**
   - **YOLO v3 Configuration File:** [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - **YOLO v3 Class Names File:** [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
   - **YOLO v3 Weights File:** [yolov3.weights](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights)
2. Launch the application.
3. Load and Process Images:
	- Click on "Select Image" to load an image.
	- Choose the desired filter by clicking on the corresponding button in the options menu.
	- Optionally, enable object detection by clicking "Detect Objects".
4. Load and Process Videos:
	- Click on "Select Video" to load a video file.
	- The video will start processing with the selected filter and YOLO object detection.
5. Live Video:
	- Click on "Live Video" to start a live feed from your webcam. You can apply filters and perform object detection in real-time.
6. Additional Options:
	- Click "Reset" to reset the image to its original state.
	- Click "Reset Filters" to remove applied filters.

## Requirements

- Python 3.12
- OpenCV
- NumPy
- Pillow
- Tkinter

## Code Overview

YOLOModel: Handles loading and applying the YOLO object detection model.
ImageProcessor: Provides various image processing functions and applies filters to images.
VideoProcessor: Manages video loading, processing, and live video capture.
MainApp: The main Tkinter application class that integrates all functionalities and provides the GUI.

## Contributing

Feel free to fork the repository and submit pull requests with improvements or bug fixes. Please make sure to test your changes and follow the existing code style.
