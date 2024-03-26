import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyttsx3
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import base64

model_best = load_model('face_model.h5') 
engine= pyttsx3.init()
# Classes 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class EmotionDetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        if text_data == "start_emotion_detection":
            await self.emotion_detection_loop()
    async def emotion_detection_loop(self):

    # Open a connection to the webcam (0 is usually the default camera)
        cap = cv2.VideoCapture(0)
        while True:
            
            ret, frame = cap.read()
            if not ret:
                # Handle webcam error
                await self.send(text_data=json.dumps({'error': 'Webcam error'}))
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract the face region
                face_roi = frame[y:y + h, x:x + w]

                # Resize the face image to the required input size for the model
                face_image = cv2.resize(face_roi, (48, 48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = image.img_to_array(face_image)
                face_image = np.expand_dims(face_image, axis=0)
                face_image = np.vstack([face_image])

                # Predict emotion using the loaded model
                predictions = model_best.predict(face_image)
                emotion_label = class_names[np.argmax(predictions)]

                # Display the emotion label on the frame
                cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
                
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #speak the emotion
                engine.say("emotion is" + emotion_label)
                engine.runAndWait()
                
                # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Send the frame data to the connected clients
            await self.send(text_data=json.dumps({'frame': frame_data}))

            # # Display the resulting frame
            # cv2.imshow('Emotion Detection', frame)

            # # Break the loop if 'q' key is pressed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()
    # Run the real-time face detection function

