import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

from utils import Utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Models:
	@ignore_warnings(category=ConvergenceWarning)

	def model_1(self, X,y):
		log_reg_model = LogisticRegression()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
		
		model = log_reg_model.fit(X_train, y_train)

		pred_model = model.predict(X_test)
		lr_acc_score = accuracy_score(y_test, pred_model)

		utils = Utils()
		utils.model_export(model,lr_acc_score)