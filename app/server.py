import joblib
import numpy as np
import flask

from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route('/predict', methods = ['GET'])
def predict():
	
	X_test = np.array([2,1,179,2])
	prediction = model.predict(X_test.reshape(1,-1))

	context = {
		'prediction': int(prediction)
	}
	
	return render_template('prediction.html',**context)


if __name__ == '__main__':

	model = joblib.load('./models/DecisionTreeClassifier.pkl')

	app.run(port=8080)





