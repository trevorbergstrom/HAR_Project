import torch
import torch.nn as nn
import sys
sys.path.append('./Dataset')
sys.path.append('./Model')
import argparse
from torchvision import datasets, models, transforms
from charades_train_data import Charades_Train_Data


def get_model(num_classes, gpu, temporal_channels):
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
	optimizer_s = torch.optim.SGD(model_s.parameters(), lr=0.001)
	criterion_t = nn.BCEWithLogitsLoss().cuda()
	optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0.001)
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

	print('-------------Saving Model----------------')
	torch.save(model_s.state_dict(), 'spatial.pth')
	torch.save(model_t.state_dict(), 'temporal.pth')
	print('-----------SAVE MODEL COMPLETE-----------')

def start():
	
	parser = argparse.ArgumentParser(description="Two Stream ConvNet for HAR")
	parser.add_argument('--num_epochs', help='Number of epochs to train for', default=2, nargs='?', type=int)
	parser.add_argument('--save_path', help='path to save the model', default='./saved_models', nargs='?')
	parser.add_argument('--learn_rate', help='Learning rate', default=0.001, type=float, nargs='?')
	parser.add_argument('--GPU', help='running on GPU?', default=False, type=bool, nargs='?')
	parser.add_argument('--Lvalue', help='L value (number of optical frames)', default=10, type=int, nargs='?')
	args = parser.parse_args()
	d = Charades_Train_Data('./Dataset/Full_data', args.Lvalue)
	train_loader = torch.utils.data.DataLoader(dataset=d, batch_size=128, shuffle=False)
	
	model_s, model_t, criterion_s, criterion_t, optimizer_s, optimizer_t= get_model(d.num_classes, args.GPU, 2*d.L_val)
	a=1
	for epoch in range(args.num_epochs):
		print("Training...")
# 		train(iter(train_loader),model,criterion, optimizer, epoch)
		train(train_loader,model_s, model_t,criterion_s, criterion_t, optimizer_s, optimizer_t, epoch, args.GPU)
	
if __name__ == '__main__':
	start()