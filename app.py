from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np 
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview')
def preview():
    df = pd.read_csv('data/Fish.csv')
    return render_template('preview.html', df_view=df)

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        Weight = request.form['Weight']
        Length1 = request.form['Length1']
        Length2 = request.form['Length2']
        Length3 = request.form['Length3']
        Height = request.form['Height']
        Width = request.form['Width']

        sample_data = [Weight, Length1, Length2, Length3, Height, Width]
        clean_data = [float(i) for i in sample_data]

        ex1 = np.array(clean_data).reshape(1,-1)

        pmodel = joblib.load('data/logit_model.pkl')
        result_prediction = pmodel.predict(ex1)
    return render_template('predict.html', Weight=Weight, Length1=Length1, Length2=Length2, Length3=Length3, Height=Height, Width=Width, result_prediction=result_prediction
                           )

if __name__ == '__main__':
    app.run(debug=True)
