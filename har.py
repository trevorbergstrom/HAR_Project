import torch
import torch.nn as nn
import sys
sys.path.append('./Dataset')
sys.path.append('./Model')
import argparse
from torchvision import datasets, models, transforms
from charades_train_data import Charades_Train_Data


def get_model(num_classes, temporal_channels=2):
	model_s = models.resnet18(pretrained=True)
	num_ftrs = model_s.fc.in_features
	model_s.fc = nn.Linear(num_ftrs, num_classes)
	
	model_t = models.resnet18(pretrained=True)
	model_t.fc = nn.Linear(num_ftrs, num_classes) 
	model_t.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3)
	
	
	criterion_s = nn.CrossEntropyLoss().cuda()
	optimizer_s = torch.optim.SGD(model_s.parameters(), lr=0.001)
	criterion_t = nn.CrossEntropyLoss().cuda()
	optimizer_t = torch.optim.SGD(model_s.parameters(), lr=0.001)
	return model_s, model_t, criterion_s, criterion_t, optimizer_s,optimizer_t

def train(train_loader,model_s,model_t,criterion_s,criterion_t, optimizer_s, optimizer_t, epoch):
	model_s.train()
	model_t.train()
	optimizer_s.zero_grad()
	optimizer_t.zero_grad()
	running_loss_s = 0.0
	running_loss_t = 0.0
	for inputs,temporal, labels in train_loader:
		output_s = model_s(inputs.float())
		loss_s = criterion_s(output_s, torch.max(labels.long(), 1)[1])
		loss_s.backward()
		optimizer_s.step()
		running_loss_s += loss_s.item()
		
		output_t = model_t(temporal.float())
		loss_t = criterion_t(output_t, torch.max(labels.long(), 1)[1])
		loss_t.backward()
		optimizer_t.step()
		running_loss_t += loss_t.item()
		
	print(f"Epoch {epoch} is spatial loss: {round(running_loss_s,4)} temporal loss: {round(running_loss_t,4)}")

def start():
	
	parser = argparse.ArgumentParser(description="Two Stream ConvNet for HAR")
	parser.add_argument('--num_epochs', help='Number of epochs to train for', default=2, nargs='?', type=int)
	parser.add_argument('--save_path', help='path to save the model', default='./saved_models', nargs='?')
	parser.add_argument('--learn_rate', help='Learning rate', default=0.001, type=float, nargs='?')
	args = parser.parse_args()
	d = Charades_Train_Data('./Dataset/Mini_data')
	train_loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False)
	
	model_s, model_t, criterion_s, criterion_t, optimizer_s, optimizer_t= get_model(d.num_classes, 2*d.L_val)
	a=1
	for epoch in range(args.num_epochs):
# 		train(iter(train_loader),model,criterion, optimizer, epoch)
		train(train_loader,model_s, model_t,criterion_s, criterion_t, optimizer_s, optimizer_t, epoch)
	
if __name__ == '__main__':
	start()