import numpy as np
import cv2 as cv
import os
from time import time
from datetime import datetime
from dotenv import load_dotenv

from threading import Thread
from queue import Queue, Empty

import re
import random
import logging

from shutil import rmtree
from statistics import mean
from math import log

#####
#
# Face Detection
#
#####

# Constants and Configuration
MIN_WIDTH = 300
MIN_HEIGHT = 300
IMG_EXT = ".png"
IMG_TO_SKIP = 5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def generate_image_directory():
    '''
    Create a unique directory for a sequence of frames.
    Returns the directory path.
    '''
    directory_name = "bestface_" + str(int(datetime.now().timestamp()))
    path = os.path.join(os.getcwd(), directory_name)
    try:
        os.mkdir(path)
        logging.info(f"Directory created at: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory: {e}")
    return path

def load_cascades():
    '''
    Load Haar cascades for face and eye detection from environment variables.
    '''
    face_cascade = cv.CascadeClassifier(os.getenv("FACE_CASCADE_CLASSIFIER", 0))
    eye_cascade = cv.CascadeClassifier(os.getenv("EYE_CASCADE_CLASSIFIER", 0))
    return face_cascade, eye_cascade

def viola_jones(img, face_cascade):
    """
    Detect a face in an image using the provided face_cascade.
    Returns a tuple: (detected, img_with_rectangle).
    """
    detected = False
    try:
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            if w >= MIN_HEIGHT and h >= MIN_HEIGHT:
                detected = True
                # Uncomment the following line if you want to draw the rectangle
                # cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    except Exception as e:
        logging.error(f"Error during face detection: {e}")
    return detected, img

def score_eyes(img, face_cascade, eye_cascade):
    """
    Check the characteristics of the eyes in the image and return a score.
    The score value is between 0 and 1, where values near 1 indicate aligned eyes.
    """
    global score_eyes_img_counter
    scores = {"size": 0, "alignment": 0}

    try:
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) != 1:
            logging.warning(f"Detected {len(faces)} faces, expected 1.")
            return 0

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi)
        if len(eyes) < 2:
            return 0

        score_eyes_img_counter += 1
        e_left, e_right = eyes[:2]

        # Size
        if e_left[2] < 60 or e_right[2] < 60:
            return 0
        scores["size"] = min(e_left[2], e_right[2]) / max(e_left[2], e_right[2])

        # Alignment
        abs_height = abs(e_left[1] - e_right[1])
        mean_eye_height = mean([e_left[2], e_right[2]])
        height_ratio = 1 - (abs_height / mean_eye_height)
        scores['alignment'] = max(height_ratio, 0)

        # Save images with eyes marked (for debugging or analysis)
        img_path = os.path.join(os.getcwd(), "eyes", f"{score_eyes_img_counter}.png")
        cv.rectangle(roi, (e_left[0], e_left[1]), (e_left[0]+e_left[2], e_left[1]+e_left[3]), (0, 255, 0), 2)
        cv.rectangle(roi, (e_right[0], e_right[1]), (e_right[0]+e_right[2], e_right[1]+e_right[3]), (255, 0, 0), 2)
        cv.imwrite(img_path, roi, [int(cv.IMWRITE_PNG_COMPRESSION), 9])

    except Exception as e:
        logging.error(f"Error during eye scoring: {e}")

    return mean(scores.values())

#####
#
# Face Selection
#
#####

def get_bestface_dirs():
    '''
    Get directories that match the "bestface_xxx" pattern.
    '''
    current_dir = os.getcwd()
    dirs = [d for d in os.listdir(current_dir) if os.path.isdir(d)]
    bestface_dirs = [d for d in dirs if re.match(r"^bestface_\d{5,10}$", d)]
    return bestface_dirs

def face_score(picture_path, face_cascade, eye_cascade):
    """
    Assign a score to a picture using its path, focusing on eye symmetry.
    Returns a score between 0 and 1.
    """
    picture = cv.imread(picture_path)
    scores = [score_eyes(picture, face_cascade, eye_cascade)]
    return mean(scores)

def select_face(face_cascade, eye_cascade):
    """
    For each directory named "bestface_xxx", analyze and save the best picture.
    """
    logging.info("Start face selection...")

    # Create directory for storing selected faces
    select_face_dir = "bestfaces"
    os.makedirs(select_face_dir, exist_ok=True)

    bestface_dirs = get_bestface_dirs()
    for d in bestface_dirs:
        logging.info(f"Processing directory: {d}")

        ranked_pictures = []

        # Get all pictures
        dir_files = os.listdir(os.path.join(os.getcwd(), d))
        pictures = [f for f in dir_files if re.match(r"^img\d{1,3}\.png$", f)]

        # Score each picture
        for pic in pictures:
            score = face_score(os.path.join(d, pic), face_cascade, eye_cascade)
            logging.info(f"Picture: {pic}\tScore: {score}")
            ranked_pictures.append({"score": score, "picture": pic})

        # Sort by score
        sorted_pictures = sorted(ranked_pictures, key=lambda entry: entry["score"], reverse=True)
        logging.info(f"Sorted pictures: {sorted_pictures}")

        # Save the best picture in another folder
        try:
            best_pic = sorted_pictures[0]["picture"]
            select_face_path = os.path.join(select_face_dir, f"{d}.png")
            os.rename(os.path.join(d, best_pic), select_face_path)
        except IndexError:
            logging.warning(f"No valid pictures in directory: {d}")
        except FileExistsError:
            os.remove(select_face_path)
            os.rename(os.path.join(d, best_pic), select_face_path)

        # Optionally delete the directory after processing
        # rmtree(d)

#####
#
# Producer / Consumer pattern
#
#####

queue = Queue(20)
run = True
score_eyes_img_counter = 0

class CameraProducerThread(Thread):
    '''
    This class provides all the images of a face when detected.
    '''
    def run(self):
        logging.info("Producer thread started")
        global queue, run
        face_cascade, _ = load_cascades()

        try:
            cap = cv.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, vj = viola_jones(gray, face_cascade)
                if ret:
                    queue.put(vj)
        except Exception as e:
            logging.error(f"CameraProducerThread error: {e}")
        finally:
            cap.release()
            logging.info("Producer thread stopped")

class PhotoTakerConsumerThread(Thread):
    '''
    This class saves a defined number of pictures.
    '''
    def run(self):
        logging.info("Consumer thread started")
        global queue, run, img_skipped
        sequence_running = False
        dir_path = None
        img_num = 0

        while run:
            try:
                img = queue.get_nowait()

                if img_skipped % IMG_TO_SKIP != 0:
                    img_skipped += 1
                    continue

                if not sequence_running:
                    sequence_running = True
                    dir_path = generate_image_directory()

                img_path = os.path.join(dir_path, f"img{img_num}{IMG_EXT}")
                cv.imwrite(img_path, img, [int(cv.IMWRITE_PNG_COMPRESSION), 9])

                img_num += 1
                queue.task_done()
            except Empty:
                sequence_running = False
                img_num = 0
            except Exception as e:
                logging.error(f"PhotoTakerConsumerThread error: {e}")
        logging.info("Consumer thread stopped")

#####
#
# Main
#
#####

if __name__=="__main__":
    face_cascade, eye_cascade = load_cascades()

    producer = CameraProducerThread()
    consumer = PhotoTakerConsumerThread()

    producer.daemon = True
    consumer.daemon = True

    # Start the threads
    producer.start()
    consumer.start()

    # Join the threads (optional)
    producer.join()
    consumer.join()

    select_face(face_cascade, eye_cascade)
