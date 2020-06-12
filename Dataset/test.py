from charades_train_data import Charades_Train_Data
from charades_test_data import Charades_Test_Data
import torch
import os
import math
import random
import itertools 
import shutil

print(os.getcwd())
train = Charades_Train_Data('./Full_data', 2)
#d = Charades_Test_Data('./Full_data', 5, 25)

print(len(train))

loader = torch.utils.data.DataLoader(dataset=train, batch_size=1, shuffle=False)

data_labels = {}
print(os.getcwd())
fn = './Full_data/Charades_v1_classes.txt'
f = open(fn, 'r')

for line in f:
	l = line.split(' ',1)
	data_labels[l[0]] = []

for clip in train.annotations:
	for action in clip.action_list:
		data_labels[action.label].append(clip.clip_name)

d_i = list(data_labels.values())
new_num = []

for i in range(len(d_i)):
	new_num.append(len(d_i[i]) / 20)	
	print(str(i) + ' ' + str(len(d_i[i])) + ' ' + str(math.floor(new_num[i])))

new_clips = []
for i in range(len(d_i)):
	c = random.choices(d_i[i], k=int(new_num[i]))
	#print(c)
	new_clips.append(c)

new_clips = list(itertools.chain.from_iterable(new_clips))
nc = []
for i in new_clips:
	if i not in nc:
		nc.append(i)
print(len(new_clips))
new_clips = nc
print(len(new_clips))
print(new_clips)

source_o = './Full_data/optical_flow/'
source_f = './Full_data/RGB_frames/'

dest_o = './Select_Data/optical_flow/'
dest_f = './Select_Data/RGB_frames/'

for f in new_clips:
	shutil.move(source_o+f, dest_o)
	shutil.move(source_f+f, dest_f)
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
