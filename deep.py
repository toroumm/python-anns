import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import sys, os

from rbm import RBM
from mlp import MLP
#from face import FaceRecognizer
from dbn import DBN


def gbrbm(inputs, neurons, epochs):

	weights = np.random.uniform(-1,1,(neurons, inputs.shape[0]))

	bias_input = np.random.uniform(-1,1,(inputs.shape[0]))

	bias_hidden = np.random.uniform(-1,1,(neurons))

	for i in xrange(epochs):

		vis = inputs[i,:]		
		
		hidden = vis * weights
		

def get_training_set(path_files):

		images = []
		
		files = os.listdir(path_files)
	
		for i in xrange(len(files)):
			if('jpg' in files[i] ):
				images.append( cv2.imread(path_files+files[i],0))
			
		return images

#**********************************************************************************************
def teste_matrix(a):

	#a = np.arange(1,10, dtype =float).reshape(3,3)

	v = np.zeros((a.shape[1]))
	m = np.zeros((a.shape[1]))
	
	#print a[0,:]
	for i in xrange(0,a.shape[1]):
		
		v[i] =  np.var(a[:,i])
		m[i] = np.mean(a[:,i])
		
		a[:,i] = (a[:,i] - np.mean(a[:,i]))/ np.var(a[:,i])
	
	
	for i in xrange(0,a.shape[1]):
		
		if(v[i] != 0):
			a[:,i] = (a[:,i] * v[i])+ m[i]
		else:
			a[:,i] = 0

	#np.nan_to_num(a)	
	#print a[0,:] 	 

def gaussian_normalize(inputs):

	for i in xrange(0,inputs.shape[1]):

		inputs[:,i] = (inputs[:,i] - np.mean(inputs[:,i])) / np.var(inputs[:,i])

	return inputs

def gaussian_desnormalize(inputs):

	for i in xrange(0,inputs.shape[1]):

		inputs[:,i] = (inputs[:,i] - np.mean(inputs[:,i])) / np.var(inputs[:,i])

	return inputs


##############################################################################################################
def plot_data(data):

	plt.xlabel('Quantidade')
	plt.ylabel('Valores')
	plt.title('Comparativo RBM x Autoencoder Backpropagation(MLP)')

	x = np.arange(0, data.shape[0])

	p1 = plt.plot(x, data[:,0])
	p2 = plt.plot(x, data[:,1])
	p3 = plt.plot(x, data[:,2])

	plt.legend([p1[0], p2[0], p3[0]], ['Original','rbm', 'mlp'])

	plt.show()

##############################################################################################################
def result(geral, size):

	tam = size*(np.sqrt(geral.shape[0]))
	
	image = np.zeros((tam,tam),dtype =int)
	
	k = 0
	for i in xrange(0, image.shape[0],size):

		for j in xrange(0, image.shape[1],size):

			a= np.reshape(geral[k,:],(size,size))
			
			image[i:i+size,j:j+size] = a[:] 
			
			#image[i*size:(i+1)*size,j*size:(j+1)*size] = a[:]
			
			k+=1
	return image.astype(int)

##############################################################################################################

def get_data(path,size):

	imgs = get_training_set(path)

	geral = np.ones((len(imgs),size*size),dtype = int)

	for i in xrange(0,len(imgs)):
		imgs[i] = cv2.resize(imgs[i],(size,size))
		a = imgs[i]
		
		
		geral[i,:] = a.flatten()

	return geral, imgs


# DBN  ##############################################################################################################


def run_dbn(geral, teste):

#epochs = np.array(([5,10,15,20,200]))

	hidden = np.array(([150]))

	epochs = np.array(([15,4]))

	ann = DBN(geral.shape[1], geral.shape[1], hidden,epochs)

	ann.training_set(geral, geral)

	ann.train_dbn()

	#ann = DBN.load('dbn_teste.pk1')
	
	dbn_output1 = result(ann.predict(geral),size)

	dbn_output2 = result(ann.predict(teste),size)

	print ann.predict(geral)

	#ann.save(ann, 'dbn_teste')

	cv2.imwrite('dbn_result1.png',dbn_output1)
	cv2.imwrite('dbn_result2.png',dbn_output2)


# RBM  ##############################################################################################################3

def run_rbm(geral,teste, neuron,learning,epochs):

	path ='/home/jeferson/Desktop/horses'

	rbm  = RBM(geral, neuron)

	rbm.set_learning_rate(learning)

	time_rbm = time.time()

	rbm.training_rbm(epochs,0)

	time_rbm = abs(time.time() - time_rbm)

	rbm_output = rbm.predict(teste)

	image = result(rbm_output, size)

	rbm.save(rbm, path+'rbm_teste_'+str(epochs))

	cv2.imwrite(path+'res_rbm_truco'+str(epochs)+'.png',image)

	print 'tempo treinamento RBM ',time_rbm, 'MSE', ((rbm_output - teste)**2).mean(axis = None), image.shape

# MLP #######################################################################################################

def run_mlp(geral,teste,neuron, learning, epochs):

	hidden = np.array(([neuron]))

	ann = MLP(geral.shape[1], geral.shape[1], hidden)

	ann.set_epochs(epochs)

	ann.set_learning_rate(learning)

	ann.set_momentum(momento)

	time_mlp = time.time()

	ann.train_mlp(geral,geral)

	time_mlp = abs(time.time() - time_mlp)

	mlp_output = ann.predict(teste)

	ann.save_mlp(ann,'mlp_teste_'+str(epochs))

	ann.plot_learning_curve()

	image = result(mlp_output, size)

	cv2.imwrite('res_mlp_2_'+str(epochs)+'.png',image)

	print 'tempo treinamento MLP', time_mlp,'MSE', ((mlp_output - teste)**2).mean(axis = None), image.shape

#Parametros e DATASET ################################################################################################

epochs = 10

learning = 0.01

momento = 0.6

size = 50

neuron = 500

path  = '/home/jeferson/Dropbox/Play_Py/faces/'

geral, imgs = get_data(path,50)


bm = RBM(geral,200)

bm.training_rbm(10)

img = bm.predict(geral)

img = result(img,50)

cv2.imwrite(path+'truco.png',img)


