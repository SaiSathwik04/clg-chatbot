
import json
import pickle
import os
import nltk
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.models import load_model

nltk.download('punkt')

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
intents = json.load(open("intents.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load sentence tokenizer using nltk default path
sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
tokenizer = TreebankWordTokenizer()

def clean_up(sentence):
    sentence_words = []
    for s in sentence_tokenizer.tokenize(sentence):
        sentence_words.extend(tokenizer.tokenize(s))
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    threshold = 0.25
    return [
        {"intent": classes[i], "probability": float(p)}
        for i, p in enumerate(result)
        if p > threshold
    ]

def get_response(intent_list):
    if intent_list:
        tag = intent_list[0]["intent"]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return np.random.choice(intent["responses"])
    return "Sorry, I didn’t get that."

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message")
    intents_list = predict_class(message)
    response = get_response(intents_list)
    return jsonify({"response": response})

@app.route("/health")
def health():
    return "OK"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
