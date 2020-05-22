import torch


sys.path.append('./Dataset')

sys.path.append('./Model')

parser = argparse.ArgumentParser(description="Two Stream ConvNet for HAR")
parser.add_argument('num_epochs', help='Number of epochs to train for', default=2, nargs='?', type=int)
parser.add_argument('save_path', help='path to save the model', default='./saved_models', nargs='?')
parser.add_argument('learn_rate', help='Learning rate', default=0.001, type=float, nargs='?')
args = parser.parse_args()

for epoch in range(epochs):
	