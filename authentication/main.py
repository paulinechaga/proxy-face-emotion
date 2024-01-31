import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np



class ModelTrainer:

    def train(self):
        # The lines below initializes the training and validation generators
        train_dir = '../dataset/train'
        val_dir = '../dataset/test'
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

        # The lines below builds the convolution network architecture
        emotion_model = Sequential()

        emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Flatten())
        emotion_model.add(Dense(1024, activation='relu'))
        emotion_model.add(Dropout(0.5))
        emotion_model.add(Dense(7, activation='softmax'))
        # emotion_model.load_weights('emotion_model.h5')

        cv2.ocl.setUseOpenCL(False)

        # Type of emotions
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # The lines below compiles and train the model:
        emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6),
                              metrics=['accuracy'])
        emotion_model_info = emotion_model.fit_generator(
            train_generator,
            steps_per_epoch=28709 // 64,
            epochs=0,
            validation_data=validation_generator,
            validation_steps=7178 // 64)

        # The line below saves the model weights
        emotion_model.save_weights('emotion_model.h5')

        return emotion_model, emotion_dict


class WebcamVideoCapture:

    def __init__(self, video_source=0, width=None, height=None):

        self.emotion_model, self.emotion_dict = ModelTrainer().train()
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = width
        self.height = height

        # Get video source width and height
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # convert float to int
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int

        self.ret = False
        self.frame = None

    def process(self):
        ret = False
        frame = None
        emotiontype = None

        if self.vid.isOpened():
            # getting video stream from webcam
            ret, frame = self.vid.read()
            maxindexx = None
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # The lines below uses openCV haarcascade xml to detect the bounding boxes
                # of face in the webcam and predict the emotions
                bounding_box = cv2.CascadeClassifier(
                    'C:/Users/josep/.conda/envs/neuralNets/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Objects of various sizes are matched, and location is returned
                num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                # cropping face and placing markers to illustrate face detection
                for (x, y, w, h) in num_faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    emotion_prediction = self.emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, self.emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    maxindexx = maxindex

        self.ret = ret
        self.frame = frame
        self.emotiontype = maxindexx

    def get_frame(self):
        self.process()  # later run in thread
        return self.ret, self.frame, self.emotiontype

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class CameraFrame(tkinter.Frame):
    emojiimge = None

    # below is a dictionary of the system path to all the emojis
    emoji_dist = {0: "../emojis/angry.png", 1: "../emojis/disgusted.png",
                  2: "../emojis/fearful.png",
                  3: "../emojis/happy.png", 4: "../emojis/neutral.png",
                  5: "../emojis/sad.png",
                  6: "../emojis/surpriced.png"}

    def __init__(self, window, video_source=0, width=None, height=None):
        super().__init__(window)

        self.window = window

        # self.window.title(window_title)
        self.video_source = video_source
        self.vid = WebcamVideoCapture(self.video_source, width, height)

        # creating a canvas for the webcam feed
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # creating a canvas to display the emoji
        image = PIL.Image.open(App.emotiontype)
        self.gcanvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.emoji = PIL.ImageTk.PhotoImage(image)
        self.image_id = self.gcanvas.create_image(0, 0, anchor=tkinter.NW, image=self.emoji)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update_widget()

    def update_widget(self):
        # Get a frame from the video source
        ret, frame, emotiontype = self.vid.get_frame()

        if ret:
            # update frame with  webcame video and bounding box of facial emotion type
            self.image = PIL.Image.fromarray(frame)
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            # updating frame with new emoji
            if emotiontype is not None:
                # retrieving emoji based on facial emotion
                image2 = tkinter.PhotoImage(file=self.emoji_dist[emotiontype])
                # updating canvas photo with emoji
                self.gcanvas.photo = image2
                self.gcanvas.itemconfig(self.image_id, image=image2)

        self.window.after(self.delay, self.update_widget)


class App:
    # default emoji on startup
    emotiontype = '../emojis/neutral.png'

    def __init__(self, window, window_title):
        self.emotiontype = App.emotiontype
        self.window = window

        self.window.title(window_title)

        self.vids = []
        vid = CameraFrame(window, 0, 400, 300)
        # creating webcam feed frame
        vid.pack()
        # creating emoji frame
        vid.gcanvas.pack()
        self.vids.append(vid)

        # Create a canvas that can fit the above video source size
        self.window.mainloop()


if __name__ == '__main__':
    # Create a window and pass it to the Application object
    App(tkinter.Tk(), "Facial Emotion Recognition")