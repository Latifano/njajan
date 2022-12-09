import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

nltk.download('popular')
lemmatizer = WordNetLemmatizer()
model = load_model('model/model.h5')


intents = json.loads(open('dataset/Dataset_PA.json').read())
words = pickle.load(open('model/texts.pkl', 'rb'))
classes = pickle.load(open('model/label.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize pattern -  pisah kata menjadi array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming tiap kata - buat bentuk yang singkat kata demi kata
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 atau 1 untuk tiap kata di bag of words yang terdapat dalam kalimat


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix N kata, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # sama dengan 1 jika kata saat ini terdapat dalam posisi kosakata dataset
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter prediksi
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # urutkan berdasarkan kekuatan probabilitas
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
        # kalau tag ga ada
        # else:
        #     result = print("Apakah mau rekomendasi yang lain?")
        # i = i+1
    # print("Apakah mau rekomendasi yang lain?")
    return result


def secondgetResponse(result2):
    result2 = print("Apakah mau rekomendasi makanan yang lain?")
    return result2


def chatbot_response(msg):
    ints = predict_class(msg, model)  # prediksi tag
    res = getResponse(ints, intents)  # ambil respon berdasarkan tag-nya
    extract_ints = ints[0]['intent']
    tags = ["daging_sapi", "daging_sapi_pedas", "daging_sapi_manis", "daging_sapi_berkuah", "daging_sapi_basah", "daging_sapi_kering", "daging_sapi_pedas_berkuah", "daging_sapi_pedas_basah", "daging_sapi_manis_berkuah", "daging_sapi_manis_basah", "daging_sapi_manis_kering", "daging_sapi_pedas_kering", "daging_ayam", "daging_ayam_pedas", "daging_ayam_berkuah", "daging_ayam_basah", "daging_ayam_kering", "daging_ayam_manis", "daging_ayam_pedas_berkuah", "daging_ayam_pedas_kering", "daging_ayam_manis_berkuah", "daging_ayam_manis_kering", "daging_ayam_manis_basah", "daging_ayam_pedas_basah", "daging_kambing", "daging_kambing_kering_pedas", "daging_kambing_berkuah_pedas", "daging_kambing_berkuah_manis", "daging_kambing_kering_manis", "seafood", "seafood_pedas", "seafood_manis", "seafood_basah", "seafood_basah_pedas", "seafood_basah_manis", "seafood_kering", "seafood_kering_pedas", "seafood_kering_manis", "seafood_berkuah", "seafood_berkuah_pedas", "seafood_berkuah_manis", "Sayur_lauk", "sayur_non_lauk", "sayur_lauk_manis", "sayur_lauk_", "sayur_lauk_berkuah", "sayur_lauk_basah", "sayur_lauk_kering", "sayur_non_lauk_manis", "sayur_non_lauk_pedas", "sayur_non_lauk_kering", "sayur_non_lauk_basah", "sayur_non_lauk_berkuah", "sayur_lauk_berkuah_pedas", "sayur_lauk_berkuah_manis", "sayur_lauk_basah_pedas", "sayur_lauk_basah_manis", "sayur_lauk_kering_manis", "sayur_lauk_kering_pedas", "sayur_non_lauk_kering_manis", "sayur_non_lauk_kering_pedas", "sayur_non_lauk_basah_pedas",
            "sayur_non_lauk_basah_manis", "sayur_non_lauk_berkuah_pedas", "sayur_non_lauk_berkuah_manis", "minuman_tradisional", "minuman_modern", "minuman_tradisional_manis", "minuman_tradisional_pahit", "minuman_tradisional_asam", "minuman_modern_manis", "minuman_modern_pahit", "minuman_modern_asam", "minuman_tradisional_panas", "minuman_tradisional_dingin", "minuman_modern_panas", "minuman_tradisional_dingin_manis", "minuman_tradisional_dingin pahit", "minuman_tradisional_panas_manis", "minuman_tradisional_panas_pahit", "minuman_tradisional_panas_asam", "minuman_modern_panas_manis", "minuman_modern_panas_pahit", "minuman_modern_panas_asam", "minuman_modern_dingin_manis", "minuman_modern_dingin_pahit", "minuman_modern_dingin_asam", "jajanan_tradisional", "jajanan_tradisional_goreng", "jajanan_tradisional_goreng_manis", "jajanan_tradisional_goreng_asin", "jajanan_tradisional_kukus", "jajanan_tradisional_kukus_manis", "jajanan_tradisional_kukus_asin", "jajanan_tradisional_asin", "jajanan_tradisional_manis", "jajanan_tradisional_panggang", "jajanan_tradisional_panggang_manis", "jajanan_tradisional_panggang_asin", "jajanan_modern", "jajanan_modern_goreng", "jajanan_modern_goreng_manis", "jajanan_modern_goreng_asin", "jajanan_modern_kukus", "jajanan_modern_kukus_asin", "jajanan_modern_kukus_manis", "jajanan_modern_panggang", "jajanan_modern_panggang_manis", "jajanan_modern_panggang_asin", "jajanan_modern_asin", "jajanan_modern_manis"]

    if extract_ints in tags:
        addition = "Apakah mau rekomendasi yang lain?"
    else:
        addition = ""

    test_respon = [res, addition]

    # return res
    return test_respon  # return ke index


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
