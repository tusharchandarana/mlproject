from flask import Flask, jsonify, request,render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application

#Route for a home page 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity') or request.form.get('ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score_raw = request.form.get('reading_score')
        writing_score_raw = request.form.get('writing_score')

        if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score_raw, writing_score_raw]):
            return render_template('home.html', results='Please fill in all fields.')

        try:
            reading_score = int(reading_score_raw)
            writing_score = int(writing_score_raw)
        except (TypeError, ValueError):
            return render_template('home.html', results='Reading and writing scores must be valid numbers.')

        if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
            return render_template('home.html', results='Scores must be between 0 and 100.')

        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        prediction = float(results[0])
        return render_template('home.html', results=f'{prediction:.2f}')
    except Exception:
        return render_template('home.html', results='Prediction unavailable right now. Please try again with valid values.')

if __name__=="__main__":
    app.run(host="0.0.0.0")