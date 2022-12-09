# ------------------------ IMPORT LIBRARY ------------------------
# import necessary libraries
from keras.models import load_model
import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import json
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings("ignore")


# Buat Objek WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# import file dataset untuk pre-processing

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("dataset/Dataset_PA.json").read()
intents = json.loads(data_file)

# ------------------------ DATA PRE-PROCESSING ------------------------
# preprocessing json data
# tokenization
# nltk.download('punkt')
# nltk.download('wordnet')
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # tambah documents
        documents.append((w, intent['tag']))

        # tambah ke daftar class
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# ------------------------ TOKENIZATION ------------------------
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# pengurutan classes
classes = sorted(list(set(classes)))

# documents = kombinasi antara patterns dan intents
print(len(documents), "documents")

# classes = intents
print(len(classes), "classes", classes)

# words = semua kata / vocabulary
print(len(words), "unique lemmatized words", words)

# membuat file pickle untuk menyimpan objek python yang akan digunakan ketika memprediksi
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('label.pkl', 'wb'))


# ------------------------ EVALUATION DATA ------------------------
# buat list kosong untuk training data
training = []

# buat array kosong untuk outputnya
output_empty = [0] * len(classes)

# training set, bag of words untuk tiap kalimat
for doc in documents:
    # inisialisasi bag of words
    bag = []
    # daftar kata - kata yang di tokenisasi untuk pattern
    pattern_words = doc[0]

    # lemmatize setiap kata -> lalu buat kata dasarnya, dg tujuan untuk mewakili kata" yang berkaitan
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]

    # buat array bag of words dg label 1, jika sebuah kata cocok dan ditemukan pada pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    # output dg label '0' untuk setiap tag dan label '1' untuk tag saat ini (untuk setiap pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# acak featurenya and ubah menjadi numpy arrays
random.shuffle(training)
training = np.array(training)

# buat list train dan test
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data telah dibuat!")

# ------------------------ MODELLING ------------------------
# Buat Neural Network model untuk memprediksi respon
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Lalu compile model. Stochastic gradient descent dengan Nesterov accelerated gradient dapat memberikan hasil yang baik untuk model ini
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# fitting and simpan model dg nama model.h5
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=400, batch_size=5, verbose=1)
model.save('model.h5', hist)  # we will pickle this model to use in the future
print("\n")
print("*"*50)
print("\nModel Berhasil Dibuat!")
