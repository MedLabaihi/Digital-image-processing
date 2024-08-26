# Face Capture and Selection

## Overview

The `face_capture_and_selection.py` script captures images from a webcam, detects faces using the Viola-Jones algorithm, and selects the best face based on eye alignment and symmetry. The selected best faces are saved in a designated directory.

## Features

- **Face Detection**: Utilizes Haar cascades for detecting faces and eyes in images.
- **Image Capture**: Continuously captures images when a face is detected.
- **Face Scoring**: Scores captured images based on eye alignment and symmetry.
- **Face Selection**: Selects and saves the best image for each detected face.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Numpy (`numpy`)
- Python Dotenv (`python-dotenv`)

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/yourrepository.git
   ```

2. **Install the required Python packages:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file:**

   Create a `.env` file in the root directory of the project, and add the paths to the Haar cascades for face and eye detection:

   ```sh
   FACE_CASCADE_CLASSIFIER=/path/to/haarcascade_frontalface_default.xml
   EYE_CASCADE_CLASSIFIER=/path/to/haarcascade_eye.xml
   ```

   Make sure you have these Haar cascade XML files available on your system.

## Usage

1. **Run the script:**

   ```sh
   python face_capture_and_selection.py
   ```

2. **Operation:**

   - The script will start capturing images from your webcam.
   - Detected faces will be processed and scored based on the alignment and size of the eyes.
   - The best images for each detected face will be saved in a directory named `bestfaces`.

3. **Stopping the script:**

   - To stop the script, press `q` in the terminal where the script is running.

## File Structure

- **face_capture_and_selection.py**: The main script file.
- **bestfaces/**: Directory where the best-selected face images are stored.
- **eyes/**: Temporary directory where images with detected eyes are stored (used for debugging).
- **.env**: Environment file containing paths to Haar cascade classifiers.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure your changes are well-documented and tested.

## Contact  
For any questions or feedback regarding this project, please contact:

- **Name**: Labaihi Mohammed
- **Email**: m.labaihi@gmail.com
- **GitHub**: [Labaihi Mohammed](https://github.com/MedLabaihi)
