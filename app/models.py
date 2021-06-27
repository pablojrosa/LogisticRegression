import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_splits
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.tree import DecisionTreeClassifier

from utils import Utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Models:
	@ignore_warnings(category=ConvergenceWarning)

	def model_1(self, X,y):
		DecisionTreeClassifier_model = DecisionTreeClassifier()
		params = {'criterion': 'gini', 'splitter': 'best'}
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

		model = DecisionTreeClassifier_model.fit(X_train, y_train)

		pred_model = model.predict(X_test)

		lr_acc_score = accuracy_score(y_test, pred_model)

		utils = Utils()
		utils.model_export(model,lr_acc_score)