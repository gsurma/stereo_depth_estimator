#!/usr/bin/python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

DPI=96
DATASET = "data/1"
DATASET_LEFT = DATASET+"/left/"
DATASET_RIGHT = DATASET+"/right/"
DATASET_DISPARITIES = DATASET+"/disparities/"
DATASET_COMBINED = DATASET+"/combined/"

def process_frame(imgL, imgR, i):
	stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7);
	disparities = stereo.compute(imgL, imgR).astype(np.float32)
	fig = plt.figure(figsize=(disparities.shape[1]/DPI, disparities.shape[0]/DPI), dpi=DPI, frameon=False);
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(disparities, cmap='jet');
	plt.savefig(DATASET_DISPARITIES+str(i)+".png")
	plt.close()
	create_combined_output(imgL, imgR, i)

def create_combined_output(imgL, imgR, i):
	combined = np.concatenate((imgL, imgR, cv2.imread(DATASET_DISPARITIES+str(i)+".png")), axis=0)
	cv2.imwrite(DATASET_COMBINED+str(i)+".png", combined)

def process_dataset():
	left_images = os.listdir(DATASET_LEFT)
	right_images = os.listdir(DATASET_RIGHT)
	assert(len(left_images)==len(right_images))
	left_images.sort()
	right_images.sort()
	for i in range(len(left_images)):
		left_image_path = DATASET_LEFT+left_images[i]
		right_image_path = DATASET_RIGHT+right_images[i]
		process_frame(cv2.imread(left_image_path, cv2.IMREAD_COLOR),
		 			  cv2.imread(right_image_path, cv2.IMREAD_COLOR),
		  			  i)

def main():
	process_dataset()
  
if __name__== "__main__":
  	main()