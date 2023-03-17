from model import X
from flask import Flask,redirect,url_for,render_template,request
import csv 
import numpy as np
import pickle
import pandas as pd
from pandas.io.parsers import read_csv

# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('home.html')
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')
    return render_template('about.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')
    return render_template('dataset.html')


@app.route('/predict')
def predicthome():
    
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)

