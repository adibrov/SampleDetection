# sample detection auxillary functions

import numpy as np
import os, tifffile
import matplotlib.pyplot as plt
from scipy.fftpack import dct

def readTile(pathToDataset="../resources/img/nice_full_wing/B=0/T=0/", tileNum=0, channelNum=1):
	"""Reads a stack corresponding to a certain tile and channel in a dataset. Follows the ZenBlue naming conventions."""
	path = pathToDataset+"S="+str(tileNum)+"/C="+str(channelNum)+"/"
	arr = np.array([tifffile.imread(path+i) for i in os.listdir(path) if i.endswith(".tif")])
	return arr


def getVarianceProfile(stack):
	"""Variance z profile of a stack. First dim is supposed to be depth."""
	depth = stack.shape[0]
	var_prof= np.zeros(depth)

	for i in range(depth):
	    var_prof[i] = np.var(stack[i,:,:])
	return np.array(var_prof)

def getIntensityProfile(stack):
	"""Variance z profile of a stack. First dim is supposed to be depth."""
	depth = stack.shape[0]
	int_prof= np.zeros(depth)

	for i in range(depth):
	    int_prof[i] = np.mean(stack[i,:,:])
	return np.array(int_prof)


def plotProfiles(profiles, figure = None, axes=None, column=1):
	"""Plot profiles (e.g. acquired with the function above)"""
	numprof = profiles.shape[0]
	length = profiles[0].shape[0]

	fig,ax = plt.subplots(nrows=length, ncols=numprof, figsize=(7,50))


	for j in range(numprof):
		
		for i in range(length):
		            # plt.subplot(length,j+1,i+1)
		            # plt.title(str(i))
		            ax[i,j].plot(profiles[j][i])

	plt.tight_layout(h_pad=2)
	plt.show()


def plotProfile(profile, figure = None, axes=None, column=1):
	"""Plot profiles (e.g. acquired with the function above)"""
	length = profile.shape[0]

	fig,ax = plt.subplots(nrows=length, ncols=1, figsize=(7,length*2))


	for i in range(length):
	            ax[i].plot(profile[i])

	plt.tight_layout(h_pad=2)
	plt.show()


def dct2d(img, norm=None):
	return dct(dct(img, axis=0, norm=norm),axis=1, norm=norm)

# DCTS 
def DCTS(img):
	norm = np.linalg.norm(img)
	return -np.sum(np.abs(img/norm)*np.log2(np.abs(img)/norm))

def cropToLeftTop(img):
	x,y = img.shape
	return np.array(img[0:x//2, 0:y//2])

def getDCTSProfile(stack):
	depth = stack.shape[0]
	dcts_prof= np.zeros(depth)

	for i in range(depth):
	    dcts_prof[i] = DCTS(cropToLeftTop(dct2d(stack[i,:,:])))
	return np.array(dcts_prof)

def normalizeToOne(img):
	return img/img.sum()



def getDCTSProfileNormalized(stack):
	depth = stack.shape[0]
	dcts_prof= np.zeros(depth)

	for i in range(depth):
	    dcts_prof[i] = DCTS(normalizeToOne(cropToLeftTop(np.abs(dct2d(stack[i,:,:])))))
	return np.array(dcts_prof)

def getDCTSProfileNormalizedExperimental(stack):
	depth = stack.shape[0]
	dcts_prof= np.zeros(depth)

	for i in range(depth):
		interm = cropToLeftTop(np.abs(dct2d(stack[i,:,:])))
		interm[0:20,0:20] = np.zeros((20,20)) + 0.00000001
		dcts_prof[i] = DCTS(normalizeToOne(interm))
	return np.array(dcts_prof)
