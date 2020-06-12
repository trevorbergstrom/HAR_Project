from charades_train_data_og import Charades_Train_Data
from charades_test_data import Charades_Test_Data
import torch
import os
import math
import random
import itertools 
import shutil

print(os.getcwd())
train = Charades_Train_Data('./Select_Data', 2)
#d = Charades_Test_Data('./Full_data', 5, 25)

print(len(train))

loader = torch.utils.data.DataLoader(dataset=train, batch_size=1, shuffle=False)

'''
mini_annots = []
for i in d.annotations:
	if i.file_name == '4DZB6':
		mini_annots.append(i)

d.get_rgb_frames(mini_annots[0])
d.get_optical_frames(mini_annots[0],1)
#print(d.annotations)
s,t,l = d[0]
'''
