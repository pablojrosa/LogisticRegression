import joblib
import numpy as np
import flask

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods = ['GET'])

def predict():
	
	X_test = np.array([71,0,0,112,149,0,1,125,0,1.6,1,0,2])
	prediction = model.predict(X_test.reshape(1,-1))

	return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':

	model = joblib.load('./models/best_model.pkl')

	app.run(port=8080)





