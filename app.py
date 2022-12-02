from flask import Flask, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model_path = './model.h5'
model = tf.saved_model(model_path)

@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route("predict")
def predict():
    pass

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run()
