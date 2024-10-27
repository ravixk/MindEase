import json
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd
import qdarkstyle

# pip install SpeechRecognition
# pip install PyAudio
import speech_recognition as sr

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


videos = {
    "sad": "https://www.youtube.com/watch?v=VooHSLtp5do",
    "stressed": "https://www.youtube.com/watch?v=HM46YzlPY9E",
    "worthless": "https://www.youtube.com/watch?v=gJscrxxl_Bg",
    "depresed": "httpss://www.youtube.com/watch?v=3pNpHZ1yv3I",
    "anxious": "https://www.youtube.com/watch?v=79kpoGF8KWU",
    "sleep": "https://www.youtube.com/watch?v=FIMdEPd98xs",
    "scared": "https://www.youtube.com/watch?v=xPDphbUyFJ4",
    "death": "https://youtu.be/RgKAFK5djSk",
    "understand": "https://www.youtube.com/watch?v=QNJL6nfu__Q",
    "help": "https://www.sukoonhealth.in/",
    "suicide": "https://www.google.com/maps/search/mental+health+experts+near+me/"
}


class MentalHealthChatbotApp(QMainWindow):
    def __init__(self):
        super(MentalHealthChatbotApp, self).__init__()

        loadUi('MentalHealthChatbotApp.ui', self)

        self.speechRecogniser = sr.Recognizer()
        self.recognised_text = ''

        with open(r'.\\training\\intents.json') as f:
            data = json.load(f)

        self.df = pd.DataFrame(data['intents'])
        # self.tokenizer = Tokenizer(lower=True, split=' ')
        # self.tokenizer.fit_on_texts(self.df['patterns'])
        # self.tokenizer.get_config()

        with open('.\\training\\Tokenizer.pkl', 'rb') as t:
            self.tokenizer = pickle.load(t)

        self.Train_model = keras.models.load_model('.\\training\\mental_health_model.h5')
        self.lbl_enc = LabelEncoder()
        self.lbl_enc.fit_transform(self.df['tag'])

        self.mic_pushButton.clicked.connect(self.Speech_Input_Prediction)
        self.predict_pushButton.clicked.connect(self.Text_Input_Prediction)

    @pyqtSlot()
    def Speech_Input_Prediction(self):

        self.notification_label.setText('Say something!')
        print("Say something!")

        with sr.Microphone() as source:
            self.speechRecogniser = sr.Recognizer()
            self.speechRecogniser.adjust_for_ambient_noise(source)
            audio = self.speechRecogniser.listen(source, timeout=5)

        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`

            self.recognised_text = self.speechRecogniser.recognize_google(audio)
            self.response_textBrowser.clear()

            if self.recognised_text:
                self.model_response(self.recognised_text)
                print("Google Speech Recognition thinks you said:\n\n" + self.recognised_text)
                self.recognised_plainTextEdit.setPlainText(self.recognised_text)
            else:
                self.notification_label.setText('Nothing to Predict, type or say something !')

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            self.notification_label.setText('Could not understand audio')

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    @pyqtSlot()
    def Text_Input_Prediction(self):

        self.response_textBrowser.clear()
        input_text = self.recognised_plainTextEdit.toPlainText()

        if self.recognised_text:
            self.model_response(self.recognised_text)
        elif input_text:
            self.model_response(input_text)
        else:
            self.notification_label.setText('Nothing to Predict, type or say something !')

    def model_response(self, query):
        text = []
        txt = re.sub('[^a-zA-Z\']', ' ', query)
        txt = txt.lower()
        txt = txt.split()
        txt = " ".join(txt)
        text.append(txt)

        x_test = self.tokenizer.texts_to_sequences(text)
        x_test = np.array(x_test).squeeze()
        x_test = pad_sequences([x_test], padding='post', maxlen=18)
        y_pred = self.Train_model.predict(x_test)
        y_pred = y_pred.argmax()

        tag = self.lbl_enc.inverse_transform([y_pred])[0]
        responses = self.df[self.df['tag'] == tag]['responses'].values[0]

        random_response = random.choice(responses)

        print("you: {}".format(query))
        print("model: {}".format(random_response))

        self.response_textBrowser.clear()
        self.response_textBrowser.append(random_response)
        self.notification_label.setText('Prediction Complete !')

        if tag in videos:
            url = videos[tag]

            print(url)

            string_val = "<a href='" + url + "'>Here is a link that could help you.</a>" + "\n\n"

            # self.response_textBrowser.append('Here is a link that could help you.')
            self.response_textBrowser.append(string_val)

            # webbrowser.open_new(url)
        else:
            pass


''' ------------------------ MAIN Function ------------------------- '''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MentalHealthChatbotApp()
    window.show()
    app.exec_()
