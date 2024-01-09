from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input
import pickle
import h5py

# create Flask application
app = Flask(__name__)

# read object TfidfVectorizer and model from disk
MODEL_PATH ='model/cnn.h5'
model = load_model(MODEL_PATH)
 
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
    
 
 
@app.route('/upload') 
def upload():
	return render_template('upload.html') 

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        try:
            df = pd.read_csv(dataset, encoding='unicode_escape')
            df.set_index('Id', inplace=True)
            return render_template("preview.html", df_view=df)
        except Exception as e:
            error = f"Error processing the file: {str(e)}"
            return render_template("preview.html", error=error) 

 
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # message
            msg = request.form['message']
            msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

            # transform data
            new_text = sequence.pad_sequences((tokenizer.texts_to_sequences(msg['data'].astype('U'))), maxlen=547)

            # model prediction
            result = model.predict(new_text, batch_size=1, verbose=2)

            # convert prediction to label
            result_label = 'Fake' if result > 0.5 else 'Real'

            return render_template('index.html', prediction_value=result_label)
        except Exception as e:
            error = f"Error predicting: {str(e)}"
            return render_template('index.html', error=error)
    else:
        error = "Invalid message"
        return render_template('index.html', error=error)
@app.route('/chart') 
def chart():
	return render_template('chart.html')

if __name__ == "__main__":
    app.run(debug=True)
