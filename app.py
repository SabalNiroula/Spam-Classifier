from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
import streamlit as st
import pickle
import nltk

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if (i.isalpha()):
            y.append(i)

    stop_word = stopwords.words('english')
    punctuation = string.punctuation
    stemmer = PorterStemmer()

    new_list = []
    for i in y:
        if i in stop_word or i in punctuation:
            y.remove(i)

    for i in y:
        new_list.append(stemmer.stem(i))

    return " ".join(new_list)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'), encoding='latin1')
model = pickle.load(open('mnb1.pkl', 'rb'))

st.title("Email/SMS spam classifier")

input_msg = st.text_area("Enter a message")

if st.button('Predict'):

    # pre process
    sms_preprocess = transform_text(input_msg)

    # vectorize
    vectorized_sms = tfidf.transform([sms_preprocess])

    # predict
    result = model.predict(vectorized_sms)[0]

    if result == 0:
        st.header("Not Spam")
    else:
        st.header("Spam")
