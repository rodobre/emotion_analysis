import cv2
import tensorflow as tf
import numpy as np
import glob
import pygame
import random
import os
from tensorflow.keras.models import load_model

label_map = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral",
        }
model = load_model('./emotion_model.hdf5', compile=True)

DISPLAY = True
NO_WEBCAM = True


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = None
if NO_WEBCAM:
    img = cv2.imread('hqdefault.png')

else:
    cam = cv2.VideoCapture(0)
    s, img = cam.read()
    if not s:
        img = None
        os.exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

face_array = []
predicted_labels = []

for (x, y, w, h) in faces:
    face_array += [gray[y:y+h, x:x+w]]

def __preprocess_input(x, v2=False):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

for face in face_array:
    face = cv2.resize(face, (50, 50))
    face = __preprocess_input(face, True)
    face = np.expand_dims(face, 0)
    face = np.expand_dims(face, -1)
    predicted_class = model.predict(face)[0]
    print(predicted_class)

    labeled_emotions = { emotion: round(score, 2) for emotion, score in enumerate(predicted_class) }
    print(labeled_emotions)
    
    predicted_label = label_map[max(labeled_emotions.keys(), key=(lambda key: labeled_emotions[key]))]
    predicted_labels += [predicted_label]

print(predicted_labels)

manele_jale = glob.glob('de_jale/*.mp3')
manele_chef = glob.glob('de_chef/*.mp3')

def play_song_on_emotion(emotion):
    global manele_jale
    global manele_chef

    if emotion in ['angry', 'disgust', 'fear', 'sad']:
        sound = pygame.mixer.music.load(os.getcwd() + '\\' + manele_jale[random.randint(0, len(manele_jale))])
        pygame.mixer.music.play()
    else:
        sound = pygame.mixer.music.load(os.getcwd() + '\\' + manele_chef[random.randint(0, len(manele_chef))])
        pygame.mixer.music.play()

pygame.mixer.pre_init(44100, 16, 2, 4096)
pygame.mixer.init()

if DISPLAY:
    print("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

    cv2.imshow(predicted_labels[0], img)
    play_song_on_emotion(predicted_labels[0])
    cv2.waitKey(0)