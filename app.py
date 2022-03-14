from distutils.log import debug
from re import template
import numpy as np
import pickle
import os
from flask import Flask, request, render_template



# Load ML model
model = pickle.load(open('hd_predict.pkl', 'rb'))

# Create application
app = Flask(__name__)

# load Image form folder
#imgFold = os.path.join('templates', 'images')
#app.config['UPLOAD_FOLDER'] = imgFold


# Bind home & prediction function to URL
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    return render_template('predict.html', output=output)


if __name__ == '__main__':
#Run the application
    app.run(debug=True)
    
    