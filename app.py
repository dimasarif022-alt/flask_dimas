from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Pastikan file model.pkl dan scaler.pkl ada di folder yang sama
with open('model.pkl', 'rb') as f:
    model = pickle.load(f) 
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data dari form HTML
    data = {
        'Pregnancies': int(request.form['pregnancies']),
        'Glucose': int(request.form['glucose']),
        'BloodPressure': int(request.form['blood_pressure']),
        'SkinThickness': int(request.form['skin_thickness']),
        'Insulin': int(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['dpf']),
        'Age': int(request.form['age'])
    }
    
    # Konversi ke DataFrame
    df = pd.DataFrame([data], index=[0])
    
    # Transformasi data menggunakan scaler
    X = scaler.transform(df)
    
    # Prediksi menggunakan model
    y = model.predict(X)
    
    # Interpretasi hasil (asumsi: 1 = Diabetic, 0 = Normal)
    prediction = 'Diabetic' if int(y[0]) == 1 else 'Normal'
    
    return render_template('index.html', prediction=prediction, model_names=model_names)

if __name__ == '__main__':
    app.run(debug=True)