"""
Jeferson de Souza 03/09/2014

"""
import numpy as np
import random,os,sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle




class Neuron:

	def __init__(self, conections):

		self.weights  = np.random.normal(0,0.01,(conections))
		self.value = 0
		self.hits = 0
		

class SOM(Neuron):

	def __init__(self, nNeurons,attributes):

		self.neurons = np.empty((nNeurons), dtype = object)
		self.learningRate = random.random()
		self.sigma = 0.0000001
		self.stopCriteria = 0.000001

		for i in xrange(0, nNeurons):
			self.neurons[i] = Neuron(attributes)

	def set_learningRate(self,value):
		self.learningRate = value

	def normalize(self, inputs):
		inputs = inputs.astype('float')
	        a,b = inputs.shape
	        for i in range(0, b):
			a = np.max(inputs[:,i])
			if(abs(a) > 0):
				inputs[:,i] /= np.max(inputs[:,i])

		return inputs

	def train_som(self,inputs, epochs):

		self.initLearning = self.learningRate
		self.initSigma = self.sigma

		inputs = self.normalize(inputs)

		for i in range(0, epochs):

			
			self.run(inputs)

			self.learningRate = self.initLearning * np.exp(-1*(i/epochs))
			self.sigma = self.initSigma * np.exp(-1*(i/epochs))

		#for j in range(0, self.neurons.size):

		#	print(j,self.neurons[j].weights[:])

	def run(self, inputs):

		a,b = inputs.shape
		aa = 0
		bb = 0
		for ii in range(0, a):

			index = self.winner(inputs[ii:ii+1])

			for i in range(0, self.neurons.size):

				if(np.sqrt(pow(self.neurons[i].value - self.neurons[index].value,2))<= self.sigma):
					self.update(i,inputs[ii:ii+1])


	def winner(self, inputs):

		win = 0
		value = 10e10
		result = np.zeros((self.neurons[0].weights.size))

		for i in range(0, self.neurons.size):

			result[:] = self.neurons[i].weights[:] - inputs[:]

	
			self.neurons[i].value = np.sqrt(np.sum(pow(result[:],2)))

			if(self.neurons[i].value < value):
				value = self.neurons[i].value
				win = i
		return win

	def update(self, index, inputs):
			self.neurons[index].weights[:] += self.gauss(index,inputs)*self.learningRate*np.sum(inputs[:] - self.neurons[index].weights[:])

	def gauss(self, index, inputs):

			return np.exp(-1*(np.sum(pow(inputs[:] - self.neurons[index].weights[:],2)) / (2*pow(self.sigma,2))))


	def predict(self, inputs):

		a, b = inputs.shape

		inputs = self.normalize(inputs)

		truco = np.zeros(a)

		for i in range(0, a):

			truco[i] = self.winner(inputs[i:i+1])

		return 	truco

	def paint_img(self, img,result):

		colors = np.zeros((self.neurons.size, 3))

		for i in range(0,self.neurons.size):
			for j in range(0,3):
				colors[i,j] = random.randint(0,255)

		a,b,c = img.shape
		k = 0
		for i in range(0, a):

			for j in range(0, b):

				img[i,j] = colors[result[k],:]
				k+=1

	def save_map(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp,save,pickle.HIGHEST_PROTOCOL)
		save.close()
		
		
	
	'''
	
		Load net (be carefull static method)
	'''	
	@staticmethod
	def load_map( path_and_namefile):

		#path_and_namefile += ".pk1"
		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)


			return mlp
		except:
			print ("Open file error, try again")



#class Lattice (SOM):
#	self __init__(som_net):
		

			
		


#***********************************************************************************************************


#
# im1 = cv2.imread("/home/jeferson/Desktop/GramaAsfalto2.png")
#
# im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
#
# im2 = cv2.imread("/home/jeferson/Desktop/VideoEditado_01.png")
#
# vtr = np.copy(im2)
#
# im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
#
# im1 = im1.reshape(im1.size,1)
#
# im2 = im2.reshape(im2.size,1)
#
# a,b,c = vtr.shape

#im1 = im1.reshape(a,b)
#im1 = np.random.random((50,1))
#im2 = np.random.random((50,1))






