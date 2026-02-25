from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and feature order
model = pickle.load(open('model.pkl', 'rb'))
feature_columns = pickle.load(open('features.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('wqp.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[col]) for col in feature_columns]
        final_features = np.array([features])
        prediction = model.predict(final_features)

        result = "Good Quality 🍷" if prediction[0] == 1 else "Bad Quality"

        return render_template('wqp.html', prediction_text=result)

    except Exception as e:
        return render_template('wqp.html', prediction_text="Error in input values.")

if __name__ == "__main__":
    app.run(debug=True)
