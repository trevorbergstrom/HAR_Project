import torch
import torchvision

# A torchvision pretrained model might work here but the closest is AlexNet which does not include dropout and has different convolutional filter sizes

# Only use if we need to create our own network...
'''
class ConvNet(nn.Module):
	def __init__(self):
		super(HO_RCNN, self).__init__()

	def forward(self,x):

;
'''

