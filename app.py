import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer  

ps = PorterStemmer()



def transform_text(text): 
    text = text.lower()
    text = nltk.word_tokenize(text) # split in words
    y = []
    for i in text :  # removing special characters
        if i.isalnum(): 
            y.append(i)
    text = y[:] # we cannot copy list directly since it is a mutable datatype
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email spam Classifier")
input_mail  = st.text_area("Enter the mail")

if st.button('Predict'):
    # Preprocess
    transformed_mail = transform_text(input_mail)
    
    
    # vectorize
    vector_input = tfidf.transform([transformed_mail])
    
    # predict
    result = model.predict(vector_input)[0]
    
    # Display
    if result == 1:
       st.header("spam")
    else:
        st.header("Not Spam")   