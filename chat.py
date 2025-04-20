import json
import pickle
import random

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.models import load_model

# Load resources
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
intents = json.load(open("intents.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Tokenizers
with open("C:/Users/saisa/nltk_data/tokenizers/punkt/english.pickle", "rb") as f:
    sentence_tokenizer = pickle.load(f)
word_tokenizer = TreebankWordTokenizer()


# Helper functions
def clean_up_sentence(sentence):
    sentence_words = []
    for sent in sentence_tokenizer.tokenize(sentence):
        sentence_words.extend(word_tokenizer.tokenize(sent))
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        for intent in intents_json["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "I'm not sure I understand. Can you rephrase?"


def chatbot_response(msg):
    intents_list = predict_class(msg)
    return get_response(intents_list, intents)


# Chat loop
print("🤖 MREC Chatbot is ready! (Type 'quit' to exit)")
while True:
    message = input("You: ")
    if message.lower() in ["quit", "exit"]:
        print("👋 Goodbye!")
        break
    response = chatbot_response(message)
    print("Bot:", response)
