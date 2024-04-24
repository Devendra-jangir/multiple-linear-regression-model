from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form inputs
        rd_spend = float(request.form['R&D Spend'])
        administration = float(request.form['Administration'])
        marketing_spend = float(request.form['Marketing Spend'])
        state = float(request.form['State'])

        # Scale the inputs
        inputs = scaler.transform([[rd_spend, administration, marketing_spend, state]])

        # Make the prediction
        prediction = model.predict(inputs)

        # Pass the prediction result to the HTML page
        return render_template('home.html', result="The predicted profit is {}".format(np.round(prediction[0])))
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
