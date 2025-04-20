import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Set up NLTK paths
nltk_data_path = "C:/Users/saisa/nltk_data"
nltk.data.path.append(nltk_data_path)

# Load sentence tokenizer manually
with open("C:/Users/saisa/nltk_data/tokenizers/punkt/english.pickle", "rb") as f:
    sentence_tokenizer = pickle.load(f)

word_tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json") as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Tokenize manually
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences = sentence_tokenizer.tokenize(pattern)
        word_list = []
        for sent in sentences:
            word_list.extend(word_tokenizer.tokenize(sent))
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and clean
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle and split
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile and train
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")

print("✅ Chatbot training complete!")
