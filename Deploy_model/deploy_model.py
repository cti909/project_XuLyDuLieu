# -------------------------------------------
# run code: cmd ->  python api_model.py
# method: POST
# api: http://127.0.0.1:5000/api/model/bert/predict
# post data: 
# {
#     "title": "Plasmonics for advance single-molecule fluorescence spectroscopy and imaging in biology",
#     "abstract": "The elucidation of complex biological processes often requires monitoring the dynamics and spatial organization of multiple distinct proteins organized on the sub-micron scale. This length scale is well below the diffraction limit of light, and as such not accessible by classical optical techniques. Further, the high molecular concentrations found in living cells, typically in the micro- to mili-molar range, preclude single-molecule detection in confocal volumes, essential to quantify affinity constants and protein-protein reaction rates in their physiological environment. To push the boundaries of the current state of the art in single-molecule fluorescence imaging and spectroscopy, plasmonic materials offer encouraging perspectives. From thin metallic films to complex nano-antenna structures, the near-field electromagnetic coupling between the electronic transitions of single emitters and plasmon resonances can be exploited to expand the toolbox of single-molecule based fluorescence imaging and spectroscopy approaches. Here, we review two of the most current and promising approaches to study biological processes with unattainable level of detail. On one side, we discuss how the reduction of the fluorescence lifetime of a molecule as it approaches a thin metallic film can be exploited to decode axial information with nanoscale precision. On the other, we review how the tremendous progress on the design of plasmonic antennas that can amplify and confine optical fields at the nanoscale, powered a revolution in fluorescence correlation spectroscopy. Besides method development, we also focus in describing the most interesting biological application of both technologies."
# }
# reponse data (success):
# {
#     "data": {
#         "predict_code": 0,
#         "predict_message": "biology"
#     },
#     "message": "Predict success",
#     "status": 200
# }
# reponse data (error):
# {
#     "error": "'content'",
#     "status": 500
# }
# -------------------------------------------
from flask import Flask, jsonify, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from keras.models import load_model
import string as s
import warnings
import re

app = Flask(__name__)
app.secret_key = "secret_key"
app.config["SECRET_KEY"] = "super-secret-key"
warnings.filterwarnings("ignore")

lemmatizer = nltk.stem.WordNetLemmatizer()
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=6)
# Load your trained model
model.load_weights('tf_model.h5')

def tokenization(text):
    lst = text.split()
    return lst

def lowercasing(lst):
    new_lst = []
    for i in lst:
        i = i.lower()
        new_lst.append(i)
    return new_lst

def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            i = i.replace(j, '')
        new_lst.append(i)
    return new_lst

def remove_numbers(lst):
    nodig_lst = []
    new_lst = []

    for i in lst:
        for j in s.digits:
            i = i.replace(j, '')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i != '':
            new_lst.append(i)
    return new_lst

def remove_stopwords(lst):
    stop = stopwords.words('english')
    new_lst = []
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

def lemmatzation(lst):
    new_lst = []
    for i in lst:
        i = lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst

def data_preprocessing(lst):
    lst = tokenization(lst)
    lst = lowercasing(lst)
    lst = remove_punctuations(lst)
    lst = remove_numbers(lst)
    lst = remove_stopwords(lst)
    lst = lemmatzation(lst)
    lst = ''.join(i + ' ' for i in lst)
    return lst


@app.route("/api/model/bert/predict", methods=['POST'])
def predict_bert():
    try:
        # input
        title = request.json['title']
        abstract = request.json['abstract']
        content = title + abstract
        print(content)
        
        # process
        result = data_preprocessing(content)
        print(result)
        
         # # Tokenize and encode the text
        inputs = tokenizer(
            result,
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        # Make predictions
        outputs = model(inputs)
        logits = outputs.logits

        # Convert logits to probabilities
        probabilities = tf.nn.sigmoid(logits)
        label_category = ['biology', 'chemistry', 'computer_science', 'mathematics', 'physics', 'economics']
        # Get the predicted label
        array = (probabilities > 0.42).numpy().astype(int)

        flattened_array = [item for sublist in array for item in sublist]
        tmp = 0
        # In ra chỉ mục của các vị trí chứa số 1
        for index, value in enumerate(flattened_array):
            if value == 1:
                tmp = index
                print("Index:", index)
                break

        label_category = ['biology', 'chemistry', 'computer_science', 'mathematics', 'physics', 'economics']
        print("Predicted Labels:", label_category[index])
        
        # news_content = preprocess(news_content)
        # news_content = tokenize_text(news_content)
        # word_ids = np.array([word_to_ids(news_content)]) # (1,512)
        # print(word_ids.shape)
        # y_pred = model_cnn_keras.predict(word_ids)
        # y_pred = y_pred.tolist()[0][0]
        # predict_message = ""
        # # y_pred in 0, 1
        # if y_pred <= 0.5: 
        #     predict_message = "Real news" 
        # else: 
        #     predict_message = "Fake news"
        reponse_data = {
            "status": 200,
            "message": "Predict success",
            "data": {
                "predict_code": index,
                "predict_message": label_category[index]
            },
        }
        return jsonify(reponse_data)
    except Exception as e:
        return jsonify({"status": 500, "error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)
