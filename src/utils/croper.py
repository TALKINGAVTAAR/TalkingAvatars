import os
import cv2
import numpy as np
from PIL import Image
import torch
from src.face3d.extract_kp_videos_safe import KeypointExtractor
from facexlib.alignment import landmark_98_to_68

class Preprocesser:
    def __init__(self, device='cuda'):
        self.predictor = KeypointExtractor(device)

    def get_landmark(self, img_np):
        """Extract 68 facial landmarks from the input image."""
        with torch.no_grad():
            dets = self.predictor.det_net.detect_faces(img_np, 0.97)

        if len(dets) == 0:
            return None

        det = dets[0]
        img = img_np[int(det[1]):int(det[3]), int(det[0]):int(det[2]), :]
        lm = landmark_98_to_68(self.predictor.detector.get_landmarks(img))
        lm[:, 0] += int(det[0])
        lm[:, 1] += int(det[1])

        return lm

    def align_face(self, img, lm, output_size=1024):
        """Align face based on landmarks."""
        lm_eye_left = lm[36:42]
        lm_eye_right = lm[42:48]
        lm_mouth_outer = lm[48:60]

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        else:
            rsize = (img.size[0], img.size[1])

        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            max(int(np.floor(min(quad[:, 0])) - border), 0),
            max(int(np.floor(min(quad[:, 1])) - border), 0),
            min(int(np.ceil(max(quad[:, 0])) + border), img.size[0]),
            min(int(np.ceil(max(quad[:, 1])) + border), img.size[1])
        )

        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            quad -= crop[0:2]

        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)
        ly = max(min(quad[1], quad[7]), 0)
        rx = min(max(quad[4], quad[6]), img.size[0])
        ry = min(max(quad[3], quad[5]), img.size[1])

        return rsize, crop, [lx, ly, rx, ry]

    def crop_face(self, image_path, save_path=None):
        """
        Detects and crops the largest face from the given image using OpenCV's Haar Cascade.
        
        Args:
            image_path (str): Path to the input image.
            save_path (str, optional): If provided, saves the cropped face to this path.
        
        Returns:
            cropped (np.ndarray): The cropped face image as a numpy array.
        """
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            raise ValueError("No faces detected.")

        # Crop the first face found
        x, y, w, h = faces[0]
        cropped = img[y:y+h, x:x+w]

        if save_path:
            cv2.imwrite(save_path, cropped)

        return cropped

    def crop(self, img_np_list, still=False, xsize=512):
        """Crop and align images in the list."""
        img_np = img_np_list[0]
        lm = self.get_landmark(img_np)

        if lm is None:
            raise RuntimeError('Cannot detect the landmark from source image')

        rsize, crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
        
        # Use crop_face to crop the largest face
        cropped_face = self.crop_face(img_np_list[0])

        # Returning cropped face and alignment details
        return [cropped_face], (rsize, crop, quad), lm
