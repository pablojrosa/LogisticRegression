import pandas as pd
import numpy as np

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from utils import Utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Models:

	def __init__(self):
		self.reg = {
		'LogisticRegression': LogisticRegression()
		}

		"""Los parametros necesitan un ser un disccionario, ya que GridSearch va a 
		recibir por parmetro un diccionario con el cual va a optimizar el modelo
		que le pasemos"""

		self.params = {
		'LogisticRegression': {
			'solver': ['lbfgs', 'liblinear']
			
			}
		}
	@ignore_warnings(category=ConvergenceWarning)
	
	def grid_training(self, X,y):

		best_score = 999
		best_model = None

		for name, reg in self.reg.items():

			grid_regresor = GridSearchCV(estimator=reg, param_grid = self.params[name], cv=3).fit(X,y.values.ravel())
			score = np.abs(grid_regresor.best_score_)

			if score < best_score:
				bast_score= score
				best_model = grid_regresor.best_estimator_


		utils = Utils()
		utils.model_export(best_model,bast_score)





















