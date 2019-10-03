import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

class Support_Vector_Machine:

	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1:'r', -1:'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)

	def train(self, data):
		self.data = data
		opt_dict = {} # optimisation dictionarry with values { ||w||: [w,b]}
		transforms = [[1,1],[-1, 1], [-1, -1], [1, -1]]
		# we pickup some halfway values to have a start
		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)
		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)
		all_data = None # Memory clear
		step_sizes =	[self.max_feature_value * 0.1,
						self.max_feature_value * 0.01,
						self.max_feature_value * 0.001]
	def predict(self, features):
		# sign (w.x + b) = in which side of the plan are we
		classification = np.sign(np.dot(np.array(features), self.w)+self.b)
		return classification