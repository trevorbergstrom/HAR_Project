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
import random

clip_annotation = namedtuple('clip_annotation', ['clip_name', 'clip_end_frame', 'action_list'])
action_annotation = namedtuple('action_annotation', ['start_frame', 'end_frame', 'label'])

#activity_gt = namedtuple('activity_gt',['file_name', 'start_time', 'end_time', 'a_class', 'clip_length'])
#class_list = namedtuple('class_list', ['abrev', 'vector_idx', 'descrip'])

class Charades_Test_Data(data.Dataset):
	def __init__(self, root_dir, Lvalue, num_frames):
		self.frame_dir = os.path.join(root_dir,'RGB_frames')
		self.flow_dir = os.path.join(root_dir,'optical_flow')
		self.classfile = os.path.join(root_dir, 'Charades_v1_classes.txt')
		self.width = 256
		self.height = 256
		self.fps = 24
		self.conv_w = 224
		self.conv_h = 224

		self.L_val = Lvalue
		self.num_frames_test = num_frames

		annot_file = os.path.join(root_dir, 'Charades_v1_test.csv')

		self.small_data_names = os.listdir(self.frame_dir)

		self.prepare_annotations(annot_file, Lvalue*num_frames)
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

	def prepare_annotations(self, annot_file, min_end_frame):
		# Want to store the annotations per clip - each clip has a list of actions
		# Reads all the annotations from the annotation csv_file
		self.annotations = []
		clips = []
		with open(annot_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			next(csv_reader)
			for row in csv_reader:
				if len(row[9]) > 0:
					if (float(row[10]) * self.fps) > min_end_frame + 1:
						clips.append([row[0], row[9], row[10]])

		for i in clips:

			#####################################################################
			# THIS SECTION CONSTRAINS THE DATASET TO THE MINI-DATA FOR BUG-FIXING
			clip_name = i[0]
			if clip_name not in self.small_data_names:
				continue
			######################################################################

			actions = i[1].split(';')
			clip_length = i[2]
			clip_end_frame = math.ceil(float(clip_length) * self.fps)

			action_list = []

			for j in actions:
				l = j.split()
				label = l[0]
				start_frame = math.floor(float(l[1]) * self.fps)
				end_frame = min(math.ceil(float(l[2]) * self.fps), clip_end_frame) # Some action labels over run the end of the clip for whatever reason.
				action_list.append(action_annotation(start_frame, end_frame, label))

			self.annotations.append(clip_annotation(clip_name, clip_end_frame, action_list))

	def __len__(self):
		return len(self.annotations)

	def center_crop(self, img, resize_w, resize_h):
		c, w, h = img.shape
		sw = w//2 - resize_w//2
		sh = h//2 - resize_h//2
		return img[:, sw:sw+resize_w, sh:sh+resize_h];

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

	def get_frame_flow(self, clip_name, frame, max_frame, L=1):
		# Optical flow can contain more than one frame, all in range(f+L)
		
		if frame + L > max_frame:
			final_frame = max_frame
		else:
			final_frame = frame + L

		frame_range = [x for x in range(frame, final_frame)]
		
		while len(frame_range) < L:
			frame_range.append(final_frame)

		stack_frames = np.empty((1,self.conv_w,self.conv_h))

		for i in frame_range:
			frame_name_x = ('{fname}/{fname}-{number}x.jpg').format(fname=clip_name,number=str(frame).zfill(6))
			frame_name_y = ('{fname}/{fname}-{number}y.jpg').format(fname=clip_name,number=str(frame).zfill(6)) 
			frame_flow = self.open_img_flow(os.path.join(self.flow_dir, frame_name_x), os.path.join(self.flow_dir, frame_name_y))
			frame_flow = self.center_crop(frame_flow, self.conv_w,self.conv_h)
			stack_frames = np.concatenate((stack_frames, frame_flow), axis=0)
			
		return np.delete(stack_frames,0,axis=0);

	def get_frame(self, clip_name, frame):
		frame_name = ('{fname}/{fname}-{number}.jpg').format(fname=clip_name,number=str(frame).zfill(6))
		rgb_frame = self.open_img_rgb(os.path.join(self.frame_dir, frame_name))
		rgb_frame = self.center_crop(rgb_frame, self.conv_w,self.conv_h)
		return rgb_frame;

	def build_gt_vec(self, frames, action_list):

		gt_vec = np.zeros(self.num_classes)
		
		for frame in frames:
			for i in action_list:
				if frame in range(i.start_frame, i.end_frame + 1):
					gt_vec[self.classes[i.label]] = 1

		return gt_vec;

	def __getitem__(self, idx):

		clip = self.annotations[idx]
		
		# Example: 400frames - (L(temporal spacing)=10 * num_frames_test=25)
		# 250 frames to complete this sequence. So out start frames could be any from 1 - 150
		possible_range_start = [x for x in range(1, clip.clip_end_frame-(self.L_val*self.num_frames_test))]

		start_frame = random.choice(possible_range_start)

		rgb_frames = []
		optical_frames = []

		for i in range(self.num_frames_test):
			frame = start_frame+(i*self.L_val)
			rgb_frames.append(self.get_frame(clip.clip_name, frame))
			optical_frames.append(self.get_frame_flow(clip.clip_name, frame, clip.clip_end_frame, L=self.L_val))
			

		
		spatial_stream = np.stack((rgb_frames), axis=0)
		temporal_stream = np.stack((optical_frames), axis=0)
		labels = self.build_gt_vec([x for x in range(start_frame, (self.L_val*self.num_frames_test))], clip.action_list)
		l = labels.tolist()
		
		return spatial_stream, temporal_stream, labels;





