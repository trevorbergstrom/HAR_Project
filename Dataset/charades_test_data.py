# This is the dataloader for the Charades Dataset. Will load to things per index. 
# Needs to load a set of frames, and the optical flow frames for each frame.
# 
import numpy as np
import torch
import torchvision.transforms as T
import torch.utils.data as data
from PIL import Image
import csv
import os
import pickle
from collections import namedtuple
import math

activity_gt = namedtuple('activity_gt',['file_name', 'start_time', 'end_time', 'a_class', 'clip_length'])
#class_list = namedtuple('class_list', ['abrev', 'vector_idx', 'descrip'])

class Charades_Test_Data(data.Dataset):
	def __init__(self, root_dir):
		self.frame_dir = os.path.join(root_dir,'RGB_frames')
		self.flow_dir = os.path.join(root_dir,'optical_flow')
		self.classfile = os.path.join(root_dir, 'Charades_v1_classes.txt')
		self.width = 224
		self.height=224

		annot_file = os.path.join(root_dir, 'Charades_v1_test.csv')

		self.small_data_names = os.listdir(self.frame_dir)

		self.prepare_annotations(annot_file)
		self.get_classes(self.classfile)
		self.num_classes = len(self.classes)

	def get_classes(self, classfile):
		self.classes = {}
		c_file = open(classfile, 'r')
		itr = 0
		for line in c_file:
			l = line.split(' ', 1)
			self.classes[l[0]] = itr
			itr+=1

	def prepare_annotations(self, annot_file):
		# Reads all the annotations from the annotation csv_file
		self.annotations = []
		clips = []
		with open(annot_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			next(csv_reader)
			for row in csv_reader:
				if len(row[9]) > 0:
					clips.append([row[0], row[9], row[10]])

		for i in clips:
			clip_name = i[0]
			if clip_name not in self.small_data_names:
				continue
			actions = i[1].split(';')
			clip_length = i[2]
			for j in actions:
				l = j.split()
				label = l[0]
				start = l[1]
				end = l[2]
				t_activity = activity_gt(clip_name, start, end, label, clip_length)
				self.annotations.append(t_activity)

	def __len__(self):
		return len(self.annotations)

	def open_img_rgb(self, path):
		img = Image.open(path)
		img = img.resize((self.width, self.height))
		img = img.convert('RGB')
		img = np.asarray(img).transpose(-1,0,1)

		return img;

	def open_img_flow(self, path_x, path_y):
		img_x = Image.open(path_x)
		img_y = Image.open(path_y)
		img_x = img_x.resize((self.width, self.height))
		img_y = img_y.resize((self.width, self.height))
		img_x = np.asarray(img_x)
		img_y = np.asarray(img_y)

		return np.stack((img_x, img_y), axis=0);

	def get_rgb_frames(self, activity):
		#print(activity)
		frame_rate = 24

		# Some cases exist where the end time annotation surpasses the clip length...
		if float(activity.end_time) > float(activity.clip_length):
			end_time = float(activity.clip_length)
		else: 
			end_time = float(activity.end_time)

		start_time = float(activity.start_time)

		#print(start_time)
		#print(end_time)

		start_frame = math.floor(start_time * frame_rate)
		end_frame = math.ceil(end_time * frame_rate) - 1

		#print(start_frame)
		#print(end_frame)

		frame_range = list(range(start_frame, end_frame))
		frame_stack = np.empty((1, 3, self.width, self.height))

		for i in frame_range:
			file_name = ('{fname}-{number}.jpg').format(fname=activity.file_name,number=str(i+1).zfill(6))
			file_name = os.path.join(self.frame_dir, os.path.join(activity.file_name,file_name))
			img = self.open_img_rgb(file_name)
			img = np.expand_dims(img, axis=0)
			frame_stack = np.concatenate((frame_stack, img), axis=0)

		#print(len(frame_range))
		
		frame_stack = np.delete(frame_stack,0,axis=0)
		
		#print(frame_stack.shape)
		return frame_stack;

	def get_optical_frames(self, activity, num_frames):
		frame_rate = 24
		#num_frames = 1 # <------------------ REMOVE FOR MORE FRAMES IN OPTICAL FLOW, ONLY WHEN FIXED
		# Some cases exist where the end time annotation surpasses the clip length...
		if float(activity.end_time) > float(activity.clip_length):
			end_time = float(activity.clip_length)
		else: 
			end_time = float(activity.end_time)

		start_time = float(activity.start_time)

		start_frame = math.floor(start_time * frame_rate)
		end_frame = math.ceil(end_time * frame_rate) # Optical flow ends one frame from the end for some reason.

		frame_range = list(range(start_frame, end_frame-1))
		#optical_frames = []
		optical_frames = {}

		#print(frame_range)
		
		for i in frame_range: 
			file_name_x = ('{fname}-{number}x.jpg').format(fname=activity.file_name,number=str(i+1).zfill(6))
			file_name_y = ('{fname}-{number}y.jpg').format(fname=activity.file_name,number=str(i+1).zfill(6))
			file_name_x = os.path.join(self.flow_dir, os.path.join(activity.file_name, file_name_x))
			file_name_y = os.path.join(self.flow_dir, os.path.join(activity.file_name, file_name_y))

			img = self.open_img_flow(file_name_x, file_name_y)
			#optical_frames.append(img)
			optical_frames[i] = img
		
		#print('OP LEN')
		#print(len(optical_frames))

		# Next we need to, given the L number, stack frames!
		l_frame_list = []
		last_frame = math.floor(float(float(activity.clip_length)* frame_rate) - 1)

		for i in frame_range:
			l_start = i
			l_end = i+num_frames
			if l_end > last_frame:
				l_end = last_frame

			f_range = [x for x in range(l_start, l_end)]
			
			if len(f_range) == 0:
				f_range.append(last_frame)

			while len(f_range) < num_frames:
				f_range.append(last_frame)

			l_frame_list.append(f_range)

		#print(l_frame_list)

		# Create stacks of optical flow frames:
		frame_stack = np.empty((1, 2*num_frames, self.width, self.height))
		for i in range(len(l_frame_list)):
			stacked_frames = np.empty((2, self.width, self.height))
			
			for j in l_frame_list[i]:
				stacked_frames = np.stack((stacked_frames, optical_frames[j]), axis=0)
			stacked_frames = np.delete(stacked_frames,0,axis=0)
			#stacked_frames = np.expand_dims(stacked_frames, axis=0)
			#print(stacked_frames.shape)
			#print(frame_stack.shape)
			frame_stack = np.concatenate((frame_stack, stacked_frames),axis=0)

		frame_stack = np.delete(frame_stack,0,axis=0)
	
		#print(len(frame_stack))
		#print(frame_stack.shape)
		return frame_stack;

	def build_gt_vec(self, idx):
		classes = np.zeros(self.num_classes)

		classes[idx] = 1
		return classes

	def __getitem__(self, idx):

		activity = self.annotations[idx]
		#print(activity)
		label_idx = self.classes[activity.a_class]
		label = self.build_gt_vec(label_idx)

		spatial_stream = self.get_rgb_frames(activity)
		temporal_stream = self.get_optical_frames(activity,1)

		return spatial_stream, temporal_stream, label;





