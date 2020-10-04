import joblib
import numpy as np
import flask

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTAMAN PARA PRUEBAS
@app.route('/predict', methods = ['GET'])

def predict():
	X_test = np.array([51,0,2,120,295,0,0,157,0,0.6,2,0,1])

	prediction = model.predict(X_test.reshape(1,-1))
	return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':

	model = joblib.load('./models/best_model.pkl')

	app.run(port=8080)





