#!/usr/bin/python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize

DPI=96
DATASET = "data/6"
DATASET_LEFT = DATASET+"/left/"
DATASET_RIGHT = DATASET+"/right/"
DATASET_DISPARITIES = DATASET+"/disparities/"
DATASET_COMBINED = DATASET+"/combined/"
DATASET_TRAINING = DATASET+"/training/"
DATASET_VALIDATION = DATASET+"/validation/"
DATASET_TESTING = DATASET+"/testing/"


def process_frame(left, right, name):
	kernel_size = 3
	smooth_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
	smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

	window_size = 9    
	left_matcher = cv2.StereoSGBM_create(
	    numDisparities=96,
	    blockSize=7,
	    P1=8*3*window_size**2,
	    P2=32*3*window_size**2,
	    disp12MaxDiff=1,
	    uniquenessRatio=16,
	    speckleRange=2,
	    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
	)

	right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

	wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
	wls_filter.setLambda(80000)
	wls_filter.setSigmaColor(1.2)

	disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
	disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )

	wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
	wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
	wls_image = np.uint8(wls_image)

	fig = plt.figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False);
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(wls_image, cmap='jet');
	plt.savefig(DATASET_DISPARITIES+name)
	plt.close()
	create_combined_output(left, right, name)

def create_combined_output(left, right, name):
	combined = np.concatenate((left, right, cv2.imread(DATASET_DISPARITIES+name)), axis=0)
	cv2.imwrite(DATASET_COMBINED+name, combined)

def divide_into_parts(image):
	width, height = image.size
	new_width = height
	new_height = height

	c_left = (width - new_width)/2
	c_top = (height - new_height)/2
	c_right = (width + new_width)/2
	c_bottom = (height + new_height)/2
	c = image.crop((c_left, c_top, c_right, c_bottom))

	r_left = c_left+new_width
	r_top = (height - new_height)/2
	r_right = c_right+new_width
	r_bottom = (height + new_height)/2
	r = image.crop((r_left, r_top, r_right, r_bottom))
	return c, r

def process_dataset():
	left_images = [f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
	right_images = [f for f in os.listdir(DATASET_RIGHT) if not f.startswith('.')]
	assert(len(left_images)==len(right_images))
	left_images.sort()
	right_images.sort()
	for i in range(len(left_images)):
		left_image_path = DATASET_LEFT+left_images[i]
		right_image_path = DATASET_RIGHT+right_images[i]
		left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
		right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
		process_frame(left_image, right_image, left_images[i])

if __name__== "__main__":
	# process_dataset()

	# disparities = [f for f in os.listdir(DATASET_DISPARITIES) if not f.startswith('.')]
	# disparities.sort()

	lefts = [f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
	lefts.sort()

	# assert(len(disparities)==len(lefts))

	for i in range(len(lefts)):
		left_center, left_right = divide_into_parts(Image.open(DATASET_LEFT+lefts[i]))
		left_center = cv2.cvtColor(np.array(left_center.resize((256, 256), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)
		#left_right = cv2.cvtColor(np.array(left_right.resize((256, 256), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)

		# disparity_center, disparity_right = divide_into_parts(Image.open(DATASET_DISPARITIES+disparities[i]))
		# disparity_center = cv2.cvtColor(np.array(disparity_center.resize((256, 256), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)
		# disparity_right = cv2.cvtColor(np.array(disparity_right.resize((256, 256), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)

		# if i % 5 == 0:
		# 	if i % 2 == 0:
		# 		output = DATASET_VALIDATION
		# 	else:
		# 		output = DATASET_TESTING
		# else:
		# 	output = DATASET_TRAINING

		#center_concat = np.concatenate((left_center, disparity_center), axis=1)
		cv2.imwrite(DATASET_TESTING+str(i)+".png", left_center) 
		# right_concat = np.concatenate((left_right, disparity_right), axis=1)
		# cv2.imwrite(output+str(i)+str((2*(i+1)))+".png", right_concat) 


	# for i, element in enumerate(disparities):
	# 	image_center, image_right = divide_into_parts(Image.open(DATASET_DISPARITIES+element))
		
	# 	image_center = image_center.resize((256, 256), Image.ANTIALIAS)
	# 	image_center.save(DATASET_OUTPUTS+str((2*(i+1))-1)+".png")

	# 	image_right = image_right.resize((256, 256), Image.ANTIALIAS)
	# 	image_right.save(DATASET_OUTPUTS+str((2*(i+1)))+".png")


	# for i, element in enumerate(lefts):
	# 	image_center, image_right = divide_into_parts(Image.open(DATASET_LEFT+element))
		
	# 	image_center = image_center.resize((256, 256), Image.ANTIALIAS)
	# 	image_center.save(DATASET_INPUTS+str((2*(i+1))-1)+".png")

	# 	image_right = image_right.resize((256, 256), Image.ANTIALIAS)
	# 	image_right.save(DATASET_INPUTS+str((2*(i+1)))+".png")
