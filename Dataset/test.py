from dataloader import Charades_Data

d = Charades_Data('./Mini_data')
mini_annots = []
for i in d.annotations:
	if i.file_name == '4DZB6':
		mini_annots.append(i)

d.get_rgb_frames(mini_annots[0])
d.get_optical_frames(mini_annots[0],1)
#print(d.annotations)
s,t,l = d[0]