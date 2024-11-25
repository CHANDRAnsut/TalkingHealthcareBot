import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model, model_from_json

import numpy as np
import speech_recognition as sr
import pyttsx3
import time

nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    result = ''

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
def calling_the_bot(txt):
    global res
    if len(txt.strip()) < 2:  # Check if the recognized text is too short or empty
        print("Input is too short or empty.")
        engine.say("Sorry, I couldn't hear your symptoms. Please speak clearly and provide more details.")
        engine.runAndWait()
        return
    
    predict = predict_class(txt)
    res = get_response(predict, intents)

    engine.say("Found it. From our database we found that " + res)
    engine.runAndWait()
    print("Your Symptom was: ", txt)
    print("Result found in our Database: ", res)


if __name__ == '__main__':
    print("Bot is Running")

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 175)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 1.0)

    voices = engine.getProperty('voices')

    engine.say("Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
    engine.runAndWait()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source)

    audio = recognizer.recognize_google(audio)
    engine.setProperty('voice', voices[0].id)
       

    while True:
        with mic as symptom:
            print("Say Your Symptoms. The Bot is Listening")
            engine.say("You may tell me your symptoms now. I am listening")
            engine.runAndWait()

            try:
                recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
                symp = recognizer.listen(symptom)

                # Check if there is actual speech input or if the input is silent
                text = recognizer.recognize_google(symp)

                # If no speech or a very short input, consider it as silence
                if len(text.strip()) < 2:  # This is a key check for silence
                    print("Error: No significant input detected")
                    engine.say("Sorry, I couldn't hear your symptoms. Please speak clearly and provide more details.")
                    engine.runAndWait()
                    continue  # Skip this iteration if no input or too short

                engine.say("You said: {}".format(text))
                engine.runAndWait()

                engine.say("Scanning our database for your symptom. Please wait.")
                engine.runAndWait()

                time.sleep(1)
                calling_the_bot(text)
            except sr.UnknownValueError:
                print("Error: Could not understand the audio")
                engine.say("Sorry, I couldn't understand the symptoms. Please try again.")
                engine.runAndWait()
            except sr.RequestError as e:
                print(f"Error with the Google Speech Recognition service: {e}")
                engine.say("Sorry, there was an issue connecting to the recognition service. Please try again later.")
                engine.runAndWait()
            except Exception as e:
                print(f"Unexpected error: {e}")
                engine.say("Sorry, there was an unexpected error. Please try again.")
                engine.runAndWait()
            finally:
                engine.say("If you want to continue, please say Continue. Otherwise, say Please exit.")
                engine.runAndWait()

        with mic as ans:
            recognizer.adjust_for_ambient_noise(ans, duration=0.2)
            voice = recognizer.listen(ans)
            final = recognizer.recognize_google(voice)

        if final.lower() == 'no' or final.lower() == 'please exit':
            engine.say("Thank You. Shutting Down now.")
            engine.runAndWait()
            print("Bot has been stopped by the user")
            exit(0)
