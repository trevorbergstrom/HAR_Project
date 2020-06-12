import torch
import torch.nn as nn
import sys
sys.path.append('./Dataset')
sys.path.append('./Model')
import argparse
from torchvision import datasets, models, transforms
from charades_train_data import Charades_Train_Data
from charades_test_data import Charades_Test_Data
from confusion_matrix import confusion_matrix
import pathlib
import os
import numpy as np
import pickle


def get_model(num_classes, gpu, temporal_channels, l_r):
	model_s = models.resnet18(pretrained=True)
	num_ftrs = model_s.fc.in_features
	model_s.fc = nn.Linear(num_ftrs, num_classes)
	
	model_t = models.resnet18(pretrained=True)
	model_t.fc = nn.Linear(num_ftrs, num_classes) 
	model_t.conv1 = nn.Conv2d(in_channels=temporal_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
	
	if gpu == True:
		model_s.cuda()
		model_t.cuda()
	
	criterion_s = nn.BCEWithLogitsLoss().cuda()
	optimizer_s = torch.optim.SGD(model_s.parameters(), lr=l_r)
	criterion_t = nn.BCEWithLogitsLoss().cuda()
	optimizer_t = torch.optim.SGD(model_t.parameters(), lr=l_r)
	return model_s, model_t, criterion_s, criterion_t, optimizer_s,optimizer_t


def train(train_loader,model_s,model_t,criterion_s,criterion_t, optimizer_s, optimizer_t, epoch, gpu):
	model_s.train()
	model_t.train()
	optimizer_s.zero_grad()
	optimizer_t.zero_grad()
	running_loss_s = 0.0
	running_loss_t = 0.0

	batch_count = 0

	for inputs,temporal, labels in train_loader:
		batch_count += 1

		if gpu == True:
			inputs = inputs.float().cuda()
			temporal = temporal.float().cuda()
			labels = labels.cuda()
		else:
			inputs = inputs.float()
			temporal = temporal.float()

		output_s = model_s(inputs)
		loss_s = criterion_s(output_s, labels)
		loss_s.backward()
		optimizer_s.step()
		running_loss_s += loss_s.item()

		output_t = model_t(temporal)
		#loss_t = criterion_t(output_t, torch.max(labels.long(), 1)[1])
		loss_t = criterion_t(output_t, labels)
		loss_t.backward()
		optimizer_t.step()
		
		optimizer_s.zero_grad()
		optimizer_t.zero_grad()

		running_loss_t += loss_t.item()
		print(f"Epoch {epoch} : batch {batch_count} spatial loss: {round(loss_s.item(),4)} temporal loss: {round(loss_t.item(),4)}")
		
	print(f"----------------------------Epoch {epoch} spatial loss: {round(running_loss_s,4)} temporal loss: {round(running_loss_t,4)}--------------------------")

def test(test_loader, model_s, model_t, gpu, num_classes):
	model_s.eval()
	model_t.eval()
	sig = nn.Sigmoid()
	preds = []
	labels = []
	
	pickle_file_p = ''
	pickle_file_l = ''
	#pickle_file_p = 'preds.pkl'
	#pickle_file_l = 'labels.pkl'

	if pickle_file_p and pickle_file_l:
		pickle_file_p = open(pickle_file_p, 'rb')
		pickle_file_l = open(pickle_file_l, 'rb')
		preds = pickle.load(pickle_file_p)
		labels = pickle.load(pickle_file_l)
	
	else:
		example_count = 0
		test_len = len(test_loader)

		for spatial, temporal, label in test_loader:
			example_count += 1
			print(f'Clip #{example_count} / {test_len}')
			
			if gpu:
				spatial = spatial.cuda()
				temporal = temporal.cuda()

			with torch.no_grad():
				s = model_s(spatial.squeeze().float())
				t = model_t(temporal.squeeze().float())
			
			s_num = s.shape[0]
			t_num = t.shape[0]

			s = s.sum(axis=0)
			t = t.sum(axis=0)
			s = s.div(s_num)
			t = t.div(t_num)
			
			pred = torch.add(s,t)
			pred = pred.div(2)
			pred = sig(pred)
	
			if gpu == True:
				pred = pred.cpu()

			label = label.squeeze()
			labels.append(label.numpy())
			preds.append(pred.numpy())

		preds = np.array(preds).astype(np.float64)
		labels = np.array(labels).astype(np.int32)

		pickle_file_p = open('preds.pkl', 'wb')
		pickle.dump(preds, pickle_file_p)
		pickle_file_p.close()
		pickle_file_l = open('labels.pkl', 'wb')
		pickle.dump(labels, pickle_file_l)
		pickle_file_l.close()

	print(preds.shape)
	print(labels.shape)
	
	c = confusion_matrix(num_classes)
	c.mAP(labels, preds)

def start():
	
	parser = argparse.ArgumentParser(description="Two Stream ConvNet for HAR")
	parser.add_argument('--num_epochs', help='Number of epochs to train for', default=2, nargs='?', type=int)
	parser.add_argument('--save_path', help='path to save the model', default='./saved_models', nargs='?')
	parser.add_argument('--learn_rate', help='Learning rate', default=0.001, type=float, nargs='?')
	parser.add_argument('--GPU', help='running on GPU?', default=False, type=bool, nargs='?')
	parser.add_argument('--Lvalue', help='L value (number of optical frames)', default=10, type=int, nargs='?')
	parser.add_argument('--Checkpoint', help='Checkpoint folder to train from', default='', nargs='?')
	parser.add_argument('--Save_dir', help='Fodler to save model to', default='./model_saves', nargs='?')
	parser.add_argument('--test', help='Test model?', default=False, nargs='?')
	parser.add_argument('--test_L', help='num optical frames for testing', default=10, nargs='?', type=int)
	parser.add_argument('--test_frames', help='number of frames for testing', default=25, nargs='?', type=int)

	args = parser.parse_args()
	
	
	pathlib.Path(args.Save_dir).mkdir(parents=True, exist_ok=True)
	model_s_path = os.path.join(args.Save_dir,'spatial.pth')
	model_t_path = os.path.join(args.Save_dir,'temporal.pth')

	if args.test:
		d = Charades_Test_Data('./Dataset/Full_data', args.test_L, args.test_frames)
	else:
		d = Charades_Train_Data('./Dataset/Full_data', args.Lvalue)

	model_s, model_t, criterion_s, criterion_t, optimizer_s, optimizer_t= get_model(d.num_classes, args.GPU, 2*args.Lvalue, args.learn_rate)

	if args.Checkpoint:
		print('Loading model checkpoint...')
		model_s.load_state_dict(torch.load(os.path.join(args.Checkpoint, 'spatial.pth')))
		model_t.load_state_dict(torch.load(os.path.join(args.Checkpoint, 'temporal.pth')))
	
	if args.test:
		test_loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False)
		test(test_loader, model_s, model_t, args.GPU, d.num_classes)
		exit()

	
	train_loader = torch.utils.data.DataLoader(dataset=d, batch_size=128, shuffle=True)

	a=1
	for epoch in range(args.num_epochs):
		print("Training...")
# 		train(iter(train_loader),model,criterion, optimizer, epoch)
		train(train_loader,model_s, model_t,criterion_s, criterion_t, optimizer_s, optimizer_t, epoch, args.GPU)
	
	print('-------------Saving Model----------------')
	torch.save(model_s.state_dict(), model_s_path)
	torch.save(model_t.state_dict(), model_t_path)
	print('-----------SAVE MODEL COMPLETE-----------')

if __name__ == '__main__':
	start()