import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import streamlit as st

# loading the model
nlp = spacy.load('en_core_web_sm')
# Instatiating Text Blob
spacy_text_blob = SpacyTextBlob()
# Add pipeline
nlp.add_pipe(spacy_text_blob)
# Input
text = 'Hello, Its a terrible weather'

# Title of App
st.title('Sentiment app')
#Input
user_input = st.text_input("Text", text)
# User Input
doc = nlp(user_input)

# Calculating Polarity and Subjectivity
st.write('Polarity:', round(doc._.sentiment.polarity, 2))
st.write('Subjectivity:', round(doc._.sentiment.subjectivity, 2)) 