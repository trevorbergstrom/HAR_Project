from charades_train_data import Charades_Train_Data
from charades_test_data import Charades_Test_Data
import torch

d = Charades_Test_Data('./Mini_data', 5, 25)
loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False)
s,t,l = d[0]

print(type(s))
print(s.shape)
print(type(t))
print(t.shape)
print(type(l))
print(l.shape)

for s, t, l in loader:
	print('break')
	print(type(s))
	print(s.shape)
	print(type(t))
	print(t.shape)
	print(type(l))
	print(l.shape)
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