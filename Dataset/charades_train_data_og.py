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
import cv2 as cv
import matplotlib.pyplot as plt

clip_annotation = namedtuple('clip_annotation', ['clip_name', 'clip_end_frame', 'action_list', 'frames_list'])
action_annotation = namedtuple('action_annotation', ['start_frame', 'end_frame', 'label', 'frames_list'])

activity_gt = namedtuple('activity_gt',['file_name', 'start_time', 'end_time', 'a_class', 'clip_length'])
#class_list = namedtuple('class_list', ['abrev', 'vector_idx', 'descrip'])

class Charades_Train_Data(data.Dataset):
	def __init__(self, root_dir, Lvalue=1):
		self.frame_dir = os.path.join(root_dir,'RGB_frames')
		self.flow_dir = os.path.join(root_dir,'optical_flow')
		self.classfile = os.path.join(root_dir, 'Charades_v1_classes.txt')
		self.width = 256
		self.height = 256
		self.fps = 24
		self.conv_w = 224
		self.conv_h = 224
		self.L_val = Lvalue

		annot_file = os.path.join(root_dir, 'Charades_v1_train.csv')

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
		# Want to store the annotations per clip - each clip has a list of actions
		# Reads all the annotations from the annotation csv_file
		self.annotations = []
		clips = []
		with open(annot_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			next(csv_reader)
			for row in csv_reader:
				if len(row[9]) > 0:
					clips.append([row[0], row[9], row[10]])

		itr = 0
		tl = len(clips)

		for i in clips:
			#print(' ')
			itr += 1
			print('Annotations clip ' + str(itr) +'/'+str(tl))
			print(i)
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
			frames_list = []
			for j in actions:
				l = j.split()
				label = l[0]
				start_frame = math.floor(float(l[1]) * self.fps)
				end_frame = min(math.ceil(float(l[2]) * self.fps), clip_end_frame) # Some action labels over run the end of the clip for whatever reason.
				#Here is where we will reduce the frames
				reduced_frames = self.reduce_action_frames(clip_name, start_frame, end_frame)
				action_list.append(action_annotation(start_frame, end_frame, label, reduced_frames))
				frames_list += reduced_frames

				'''
				print('Action Start: ' + str(start_frame))
				print("Action End: " + str(end_frame))
				print('Num_Reduced Frames: ' + str(len(reduced_frames)))
				print('reduced_frames: ')
				print(reduced_frames)
				'''

			res_frames_list = []
			for j in frames_list:
				if j not in res_frames_list:
					res_frames_list.append(j)
			res_frames_list.sort()
			
			#print('CLIP list')
			#print(res_frames_list)
			
			self.annotations.append(clip_annotation(clip_name, clip_end_frame, action_list, res_frames_list))
		self.save_annotations()

	def save_annotations():
		pickle_file_annotations = open('annotations.pkl', 'wb')
		pickle.dump(self.annotations, pickle_file_annotations)
		pickle_file_annotations.close()

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

	def build_gt_vec(self, frame, action_list):

		gt_vec = np.zeros(self.num_classes)
		
		for i in action_list:
			if frame in i.frames_list:
				gt_vec[self.classes[i.label]] = 1
			#if frame in range(i.start_frame, i.end_frame + 1):
				#gt_vec[self.classes[i.label]] = 1

		return gt_vec;

	def __getitem__(self, idx):
		# Per the paper, we sample a single random frame from the clip
		clip = self.annotations[idx]
		#frame_range = [x for x in range(1, clip.clip_end_frame)]
		frame = random.choice(clip.frames_list)
		
		spatial_stream = self.get_frame(clip.clip_name, frame)
		temporal_stream = self.get_frame_flow(clip.clip_name, frame, clip.clip_end_frame, L=self.L_val)
		
		# TO get the label: since multiple actions can be occuring at each frame, we need to select all the labels that are present at that frame. 
		label = self.build_gt_vec(frame, clip.action_list)
		return spatial_stream, temporal_stream, label;

	def display(self,*imgs):
		n = len(imgs)
		f = plt.figure()
		for i, img in enumerate(imgs):
			f.add_subplot(1, n, i + 1)
			plt.imshow(img)
		plt.show()

	def reduce_action_frames(self, clip_name, start_frame, end_frame, threshold=0.10):
		saved_frames = []
		last_frame = 1
		for i in range(start_frame+1,end_frame):
			frame = self.get_frame(clip_name, i)
			diff = self.frame_diff(clip_name, i, last_frame)
			if diff > threshold:
				saved_frames.append(i)
				last_frame = i
		return saved_frames
	
	def frame_diff(self, clip_name, current_frame, ref_frame):
		cf = self.get_frame(clip_name, current_frame).transpose(1,2,0)
		rf = self.get_frame(clip_name, ref_frame).transpose(1,2,0)
		hsv_base = cv.cvtColor(rf, cv.COLOR_BGR2HSV)
		hsv_test1 = cv.cvtColor(cf, cv.COLOR_BGR2HSV)
		
		h_bins = 50
		s_bins = 60
		histSize = [h_bins, s_bins]
		
		# hue varies from 0 to 179, saturation from 0 to 255
		h_ranges = [0, 180]
		s_ranges = [0, 256]
		ranges = h_ranges + s_ranges # concat lists
		# Use the 0-th and 1-st channels
		channels = [0, 1]
		
		hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
		cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
		
		hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
		cv.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
		
		
# 		compare_method = 0
# 		base_base = cv.compareHist(hist_base, hist_base, compare_method)
# 		base_test1 = cv.compareHist(hist_base, hist_test1, compare_method)
# 		print('Method:', compare_method, 'Perfect, Base-Test(1):',\
# 	          base_base, '/', base_test1 )
# 		print(f'frame num {current_frame}')	
		diffs = []
		for compare_method in range(4):
		    base_base = cv.compareHist(hist_base, hist_base, compare_method)
		    base_test1 = cv.compareHist(hist_base, hist_test1, compare_method)
		    diffs.append((base_base,base_test1))
# 		    print('Method:', compare_method, 'Perfect, Base-Test(1):',\
# 		          base_base, '/', base_test1)
		return diffs[3][1]
		return base_test1

