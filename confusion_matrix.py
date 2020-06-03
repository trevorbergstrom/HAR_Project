import numpy as np
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import math

class confusion_matrix():
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.mtx = np.zeros((num_classes, 4))
		self.labels = [x+1 for x in range(self.num_classes)]
	
	def add_labels(self, labels):
		self.labels = labels
	
	def mAP(self, y_true, y_pred):
		ay_true = np.column_stack(y_true[:-1])
		ay_pred = np.column_stack(y_pred[:-1])

		y_true = np.column_stack(y_true)
		y_pred = np.column_stack(y_pred)

		avg_precision = []
		avg_precision_a = []

		# Looping for each class:
		for i in range(self.num_classes):
			print(y_true[i])
			print(y_pred[i])
			precision, recall, _ = precision_recall_curve(y_true[i], y_pred[i])
			avg_precision.append(average_precision_score(y_true[i], y_pred[i],average='weighted'))
			avg_precision_a.append(average_precision_score(ay_true[i], ay_pred[i], average='weighted'))

			#txt='Class #{c} : Precision = {p} : Recall = {r} : Avg_Precision = {a}'
			txt='Class #{c} : Avg_Precision = {a} : AP_a = {aa}'
			print(txt.format(c=self.labels[i], a=avg_precision[i], aa=avg_precision_a[i]))

		avg_precision_a1 = []
		
		for i in avg_precision_a:
			if math.isnan(i) == False:
				avg_precision_a1.append(i)

		txt = 'Final mAP [EXTRA-CLASS] = {ap} : mAP_a = {a}'
		print(txt.format(ap=(sum(avg_precision)/self.num_classes), a=(sum(avg_precision_a1) / len(avg_precision_a1))))









