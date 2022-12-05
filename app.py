import io
from flask import Flask, render_template, request, abort, send_file, redirect, jsonify, Response
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TF warning 제거

app = Flask(__name__)

model_path = 'models/model.h5'
model = tf.keras.models.load_model(model_path)

original = pd.read_csv('models/Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)
original = original.values
@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

def denormalize_series(data, min, max):
    data = data * (max - min) + min
    return data
@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file != '':
            filename = os.path.splitext(secure_filename(file.filename))[0]
            file_bytes = file.read()
            df = pd.read_csv(io.BytesIO(file_bytes),
                             infer_datetime_format=True, index_col='Week of', header=0)
            df = np.array(df).reshape(-1, 10, 1)
            forecast = model.predict(df)
            dn = denormalize_series(forecast, original.min(axis=0), original.max(axis=0))
            dn = dn.flatten().reshape(10, -1)
            dn = pd.DataFrame(dn)
            output_stream = io.StringIO()
            dn.to_csv(output_stream)
            response = Response(
                output_stream.getvalue(),
                mimetype='text/csv',
                content_type='application/octet-stream',
            )
            filename = os.path.join(filename, "result")
            response.headers["Content-Disposition"] = "attachment; filename="+filename+".csv"
            return response
    else:
        return redirect("/")

if __name__ == '__main__':
    app.run()
