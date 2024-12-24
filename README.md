# README: Fine-Tuning BERT for Multi-Class Sentiment Classification and Streamlit App

## **Project Overview**
This project demonstrates fine-tuning BERT for **multi-class sentiment classification** on Twitter tweets and integrates the trained model into a **Streamlit app** for real-time text classification.

---

## **Folder Structure**
1. **Fine_Tuning_BERT_for_Multi_Class_Sentiment_Classification_for_Twitter_Tweets.ipynb** - Jupyter Notebook for fine-tuning a BERT model.
2. **app.py** - Streamlit app for deploying the fine-tuned model.

---

## **File 1: Fine_Tuning_BERT_for_Multi_Class_Sentiment_Classification_for_Twitter_Tweets.ipynb**

### **Purpose**
This notebook is designed for **fine-tuning BERT** on a dataset of Twitter tweets to classify sentiment into multiple classes.

### **Key Steps in the Notebook**
1. **Library Imports**
   - Loads necessary libraries such as **Transformers**, **PyTorch**, and **Sklearn**.
2. **Dataset Preparation**
   - Loads and preprocesses Twitter data for training and validation.
   - Encodes text data using **tokenizers**.
3. **Model Definition**
   - Uses **BERT-base-uncased** as the pre-trained model.
   - Adds a classifier head for sentiment classification.
4. **Training Setup**
   - Splits data into training and validation sets.
   - Configures an optimizer, scheduler, and loss function.
5. **Model Training**
   - Fine-tunes BERT with evaluation on validation data.
6. **Evaluation and Metrics**
   - Computes **accuracy** and **F1-score** to evaluate model performance.
7. **Saving the Model**
   - Exports the fine-tuned model for deployment in the Streamlit app.

### **Saving the Model After Fine-Tuning**
After fine-tuning, save the model using the following command:
```python
trainer.save_model("bert-base-uncased-sentiment-model")
```
- The saved model can then be loaded directly using the Hugging Face pipeline for inference.

### **Output**
- Trained BERT model saved as **'bert-base-uncased-sentiment-model'**.

---

## **File 2: app.py**

### **Purpose**
This Python script creates a **Streamlit web application** to deploy the fine-tuned BERT model for real-time sentiment classification.

### **Code Breakdown**

1. **Imports Required Libraries**:
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   from transformers import pipeline
   import torch
   ```
   - **Streamlit** - Framework for creating web apps.
   - **Transformers** - Provides the pipeline to load the fine-tuned BERT model.
   - **Torch** - Enables GPU support for model inference.

2. **Device Setup**:
   ```python
   device = 0 if torch.cuda.is_available() else -1
   ```
   - Checks if a **GPU** is available for faster inference; falls back to **CPU** if no GPU is found.

3. **Streamlit UI Design**:
   ```python
   st.title("Fine Tuning BERT for Twitter Tweets for Multi Class Sentiment Classification")
   ```
   - Displays the title of the app.

4. **Pipeline Initialization**:
   ```python
   classifier = pipeline('text-classification', model= 'bert-base-uncased-sentiment-model', device=device)
   ```
   - Loads the fine-tuned BERT model for text classification.
   - Uses the **device** parameter to run on GPU or CPU as available.

5. **User Input and Prediction**:
   ```python
   text = st.text_area("Enter some text")

   if st.button("Predict"):
       result = classifier(text)
       st.write(result)
   ```
   - Accepts user input through a text area.
   - Predicts sentiment by passing input through the model.
   - Displays the result in real time.

### **Output**
- Displays sentiment classification results (e.g., **label** and **confidence score**) for the entered text.

---

## **How to Run the Project**

### **Step 1: Environment Setup**
1. Create a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate   # Linux/Mac
   env\Scripts\activate      # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure dependencies like **transformers**, **torch**, **sklearn**, and **streamlit** are installed.)*

### **Step 2: Model Training (Optional)**
- Open the Jupyter Notebook and run all cells to fine-tune the model.
- Save the trained model using:
  ```python
  trainer.save_model("bert-base-uncased-sentiment-model")
  ```
- The saved model can then be loaded directly using the Hugging Face pipeline.
- Ensure the model is saved with the name **'bert-base-uncased-sentiment-model'**.

### **Step 3: Launch Streamlit App**
Run the Streamlit application:
```bash
streamlit run app.py
```

### **Step 4: Test the App**
- Open the app in a browser (usually http://localhost:8501).
- Enter sample text in the input box and click "Predict".
- View sentiment classification results.

---

## **Requirements**
- Python 3.7+
- Libraries:
  - `transformers`
  - `torch`
  - `streamlit`
  - `sklearn`
  - `pandas`
  - `numpy`

---

## **Notes**
- **GPU Support**: If a GPU is available, the app will automatically use it for faster inference.
- **Model Path**: Ensure the model is saved under the name **'bert-base-uncased-sentiment-model'** in the working directory.
- **Docker Support**: For containerized deployment, include a **Dockerfile** with GPU support and dependencies.

---

## **Author**
This project is developed for sentiment classification using **BERT** and deployed via **Streamlit**.

