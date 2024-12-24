import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch


device = 0 if torch.cuda.is_available() else -1

st.title("Fine Tuning BERT for Twitter Tweets for Multi Class Sentiment Classification")

classifier = pipeline('text-classification', model= 'bert-base-uncased-sentiment-model', device=device)

text = st.text_area("Enter some text")

if st.button("Predict"):
    result = classifier(text)
    st.write(result)