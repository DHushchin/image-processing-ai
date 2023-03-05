import dlib 
import os
import cv2
from glob import glob
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(f'{current_dir}/models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(f'{current_dir}/models/dlib_face_recognition_resnet_model_v1.dat')

VALID_IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']


def get_image_paths(root_dir, class_names):
    '''Returns a list of paths to all images in the directory'''
    image_paths = []

    for class_name in class_names:
        class_dir = os.path.sep.join([root_dir, class_name])
        class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))

        for file_path in class_file_paths:
            _, extension = os.path.splitext(file_path)
            if extension in VALID_IMAGE_EXTENSIONS:
                image_paths.append(file_path)

    return image_paths


def face_rects(image):
    '''Returns a list of dlib rectangles corresponding to the detected faces'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1 should upsample the image 1 time. 
    # This will make everything bigger and allow us to detect more faces.
    return face_detector(gray, 1)


def face_landmarks(image):
    '''Returns a list of facial landmarks'''
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]
    # a list of 68 points that define the face 


def face_encodings(image):
    '''Returns a list of 128-dimensional face encodings'''
    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark, 1))
             for face_landmark in face_landmarks(image)]


def compare_faces(known_encoding, face_encoding, tolerance=0.6):
    '''Returns a list of True/False values indicating which known_encodings match the face_encoding'''
    return sum(np.linalg.norm(known_encoding - face_encoding, axis=1) <= tolerance)
