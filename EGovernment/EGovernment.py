from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import joblib
# from sklearn.externals import joblib
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
# from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing import image
import os
from numpy import dot
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import imutils
import nltk

main = tkinter.Tk()
main.title("Automating E-Government")
main.geometry("1300x1200")

mainframe=Frame(main,bg="pink")
mainframe.pack(fill='both',expand=True)

global filename
global text_sentiment_model
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
global face_detection
global image_sentiment_model
global digits_cnn_model


def digitModel():
    global digits_cnn_model
    with open('models/digits_cnn_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        digits_cnn_model = model_from_json(loaded_model_json)

    digits_cnn_model.load_weights("models/digits_cnn_weights.h5")
    # digits_cnn_model._make_predict_function()
    print(digits_cnn_model.summary())
    text.insert(END, 'Digits based Deep Learning CNN Model generated\n')


def sentimentModel():
    global text_sentiment_model
    global image_sentiment_model
    global face_detection
    text_sentiment_model = joblib.load('models/sentimentModel.pkl')
    text.insert(END, 'Text based sentiment Deep Learning CNN Model generated\n')

    face_detection = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    image_sentiment_model = load_model('models/_mini_XCEPTION.106-0.65.hdf5', compile=False)
    text.insert(END, 'Image based sentiment Deep Learning CNN Model generated\n')
    print(image_sentiment_model.summary())


def digitRecognize():
    global filename
    filename = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");

    imagetest = image.load_img(filename, target_size=(28, 28), grayscale=True)
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis=0)
    pred = digits_cnn_model.predict(imagetest.reshape(1, 28, 28, 1))
    predicted = str(pred.argmax())
    imagedisplay = cv2.imread(filename)
    orig = imagedisplay.copy()
    output = imutils.resize(orig, width=400)
    cv2.putText(output, "Digits Predicted As : " + predicted, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Predicted Image Result", output)
    cv2.waitKey(0)


def opinion():
    user = simpledialog.askstring("Please enter your name", "Username")
    opinion = simpledialog.askstring("Government Service Opinion",
                                     "Please write your Opinion about government services & policies")
    f = open("Peoples_Opinion/opinion.txt", "a+")
    f.write(user + "#" + opinion + "\n")
    f.close()
    messagebox.showinfo("Thank you for your opinion", "Your opinion saved for reviews")


def stem(textmsg):
    stemmer = nltk.stem.PorterStemmer()
    textmsg_stem = ''
    textmsg = textmsg.strip("\n")
    words = textmsg.split(" ")
    words = [stemmer.stem(w) for w in words]
    textmsg_stem = ' '.join(words)
    return textmsg_stem


def viewSentiment():
    text.delete('1.0', END)
    with open("Peoples_Opinion/opinion.txt", "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            arr = line.split("#")
            text_processed = stem(arr[1])
            X = [text_processed]
            sentiment = text_sentiment_model.predict(X)
            predicts = 'None'
            if sentiment[0] == 0:
                predicts = "Negative"
            if sentiment[0] == 1:
                predicts = "Positive"
            text.insert(END, "Username : " + arr[0] + "\n");
            text.insert(END, "Opinion  : " + arr[1] + " : Sentiment Detected As : " + predicts + "\n\n")


def uploadPhoto():
    filename = filedialog.askopenfilename(initialdir="expression_images_to_upload")
    user = simpledialog.askstring("Please enter your name", "Username")
    policy = simpledialog.askstring("Please enter Government Policy name related to Facial Expression",
                                    "Please enter Government Policy name related to Facial Expression")
    img = cv2.imread(filename)
    cv2.imwrite("sentimentImages/" + user + "-" + policy + ".jpg", img);
    messagebox.showinfo("Your facial expression image accepted for reviews",
                        "Your facial expression image accepted for reviews")


def photoSentiment():
    filename = 'sentimentImages'
    for root, dirs, files in os.walk(filename):
        for fdata in files:
            frame = cv2.imread(root + "/" + fdata)
            faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            msg = ''
            if len(faces) > 0:
                faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (x, y, w, h) = faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = temp[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = image_sentiment_model.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                msg = "Sentiment detected as : " + label
                img_height, img_width = frame.shape[:2]
                cv2.putText(frame, msg, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow(fdata, frame)
                messagebox.showinfo(fdata, "Sentiment predicted from Facial expression as : " + label)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


verticalFrame=Frame(mainframe,bg="black")
title = Label(verticalFrame, text='Automating E-Government Services With Artificial Intelligence', anchor=W,justify='center')
title.config(bg='black', fg='white')
font = ('times', 16, 'bold')
title.config(font=font)
title.config(height=3)
title.pack(padx=20,pady=10)
verticalFrame.pack(fill="x",padx=10,pady=10)

horizontal_frame=Frame(mainframe,bg="pink")

buttons_frame=Frame(horizontal_frame,bg="pink")

#button 1
font1 = ('times', 14, 'bold')
digitButton = Button(buttons_frame, text="Generate Hand Written Digits Recognition Deep Learning Model",command=digitModel)
digitButton.config(font=font1)
digitButton.pack(pady=5,fill='x')



#button 3
sentimentButton = Button(buttons_frame, text="Generate Text & Image Based Sentiment Detection Deep Learning Model",command=sentimentModel)
sentimentButton.config(font=font1)
sentimentButton.pack(pady=5,fill='x')

#button 4
recognizeButton = Button(buttons_frame, text="Upload Test Image & Recognize Digit",command=digitRecognize)
recognizeButton.config(font=font1)
recognizeButton.pack(pady=5,fill='x')

#button 5
opinionButton = Button(buttons_frame, text="Write Your Opinion About Government Policies",command=opinion)
opinionButton.config(font=font1)
opinionButton.pack(pady=5,fill='x')

#button 6
viewButton = Button(buttons_frame, text="View Peoples Sentiments From Opinions", command=viewSentiment)
viewButton.config(font=font1)
viewButton.pack(pady=5,fill='x')
#button 7
photoButton = Button(buttons_frame, text="Upload Your Face Expression Photo About Government Policies",command=uploadPhoto)
photoButton.config(font=font1)
photoButton.pack(pady=5,fill='x')
#button 8
photosentimentButton = Button(buttons_frame, text="Detect Sentiments From Face Expression Photo",command=photoSentiment)
photosentimentButton.config(font=font1)
photosentimentButton.pack(pady=5,fill='x')

#button 2
pathlabel = Label(buttons_frame)
pathlabel.config(bg='pink', fg='black')
pathlabel.config(font=font1,width=50)
pathlabel.pack(pady=5,fill='x')

buttons_frame.grid(row=0,column=0,padx=10,pady=5,sticky="nsew")
#---------------------------------------------------------------------------------------
#text frame
text_frame=Frame(horizontal_frame,bg='pink')

font1 = ('times', 12, 'bold')
text = Text(text_frame, height=15, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.config(font=font1)
text.pack()

title_bn = Label(text_frame, text='Project by \nBatch-2\nCSE-IV-A\nCMR Technical Campus', anchor=W,justify='center')
title_bn.config(bg='pink', fg='grey')
font_bn= ('times', 16, 'bold')
title_bn.config(font=font_bn)
title_bn.config(height=5)
title_bn.pack(fill='x',padx=(350,0),pady=200)




text_frame.grid(row=0,column=1,padx=10,pady=10,sticky="nsew")

horizontal_frame.grid_columnconfigure(0,weight=1)
horizontal_frame.grid_columnconfigure(1,weight=1)
horizontal_frame.pack(fill="x")

main.mainloop()

