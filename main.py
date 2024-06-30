from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import dlib
import numpy as np
import cv2
from typing import Optional

app = FastAPI()

class Profile(BaseModel):
    name: str
    averaged_profile: list
    profiles: list = []

face_library = {}

# Dlib's trained models for face detection, analysis, and recognition
detection_model = dlib.get_frontal_face_detector()
prediction_model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facial_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


@app.post("/create-profile/{id}/{name}", response_model=Profile)
async def create_profile(id: int, name: str, file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    
    # Facial analysis
    profile_description = analyze_face(img)

    # Add profile to library of faces
    if id in face_library:
        profile = face_library[id]
        profile.profiles.append(list(profile_description))

        # Compute averaged_profile when new image is uploaded
        nparray = []
        for item in profile.profiles:
            nparray.append(np.asarray(item))
        nparray = np.asarray(nparray)
        nparray = np.stack(nparray)

        profile.averaged_profile = list(np.mean(nparray, axis=0))
    else:
        profile = Profile(name=name, averaged_profile=profile_description, profiles=[list(profile_description)])
        face_library[id] = profile

    return profile


@app.post("/recognize-profile")
async def recognize_profile(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))

    return recognize_face(img)


def analyze_face(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face
    face = detection_model(grayscale)[0]

    # Extract landmarks
    landmarks = prediction_model(grayscale, face)
    face_description = np.array(facial_recognition_model.compute_face_descriptor(image, landmarks))

    return face_description


def recognize_face(image):
    face_description = analyze_face(image)

    for id, profile in face_library.items():
        distance = np.linalg.norm(np.asarray(profile.averaged_profile) - face_description)
        if distance < 0.3: 
            return face_library[id].name
        
    return None


