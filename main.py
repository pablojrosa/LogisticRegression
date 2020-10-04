from utils import Utils
from models import Models

if __name__ == '__main__':

	utils = Utils()
	models = Models()

	data = utils.load_from_csv('in/heart.csv')
	data = data[data['chol'] < data.chol.quantile(0.95)].reset_index(drop=True)

	print('pasa')

	X , y = utils.features_target(data,['thal'], ['thal'])

	models.grid_training(X,y)
