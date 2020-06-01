import torch
import torch.nn as nn
import sys
sys.path.append('./Dataset')
sys.path.append('./Model')
import argparse
from torchvision import datasets, models, transforms
from charades_train_data import Charades_Train_Data


def get_model(num_classes):
	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, num_classes)
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.001)
	return model_ft, criterion, optimizer

def train(train_loader,model,criterion, optimizer, epoch):
	model.train()
	optimizer.zero_grad()
	running_loss = 0.0
	for inputs,temporal, labels in train_loader:
		outputs = model(inputs.float())
		loss = criterion(outputs, torch.max(labels.long(), 1)[1])
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	print(f"running loss for epoch {epoch} is {round(running_loss,4)}")

def start():
	
	parser = argparse.ArgumentParser(description="Two Stream ConvNet for HAR")
	parser.add_argument('num_epochs', help='Number of epochs to train for', default=2, nargs='?', type=int)
	parser.add_argument('save_path', help='path to save the model', default='./saved_models', nargs='?')
	parser.add_argument('learn_rate', help='Learning rate', default=0.001, type=float, nargs='?')
	args = parser.parse_args()
	d = Charades_Train_Data('./Dataset/Mini_data')
	train_loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False)
	
	model, criterion, optimizer= get_model(train_loader.dataset.num_classes)
	a=1
	for epoch in range(args.num_epochs):
# 		train(iter(train_loader),model,criterion, optimizer, epoch)
		train(train_loader,model,criterion, optimizer, epoch)
	
if __name__ == '__main__':
	start()