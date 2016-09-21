import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

class Neuron:

	def __init__(self):

		self.bias = np.random.random()

		self.output = 0

		self.gradient = 0
class Map:

	def __init__(self,n_neurons,connections):

		self.weights = np.random.uniform(-0.01,0.01,size = connections)

		self.neurons = np.empty((n_neurons),dtype = object)

		self.learning_rate = 000.1

		self.hiper_a = 0.2

		self.hiper_b = 0.4

		self.func_activation = 1

		for i in xrange(0, n_neurons):

			self.neurons[i] = Neuron()

	def sigmoide(self, x):
		return 1 / (1+np.exp(-x))

	def dev_sigmoide(self,y):
		return y*(1.0-y)

	def hiperbolic(self,x): # terminar
		return x*x
		
	def dev_hipoerbolic(self, y):
		return y*y #terminar
 
	def forward(self,inputs):

		for i in xrange(0, self.neurons.shape[0]):

			if(self.func_activation == 0):
				self.neurons[i].output = self.simoide(np.dot(inputs,self.weights)+ self.neurons[i].bias)
			else:
				self.neurons[i].output = self.hiperbolic(np.dot(inputs,self.weights)+ self.neurons[i].bias)

	def backward(self, gradient,weights,inputs): # inputs  = entrada naquela conexão do neuronio, ou seja, a saída da camada anterior
			
		for i in xrange(0, self.neurons.shape[0]):

			if(self.func_activation == 0):
				self.neurons[i].gradient = self.dev_simoide(np.dot(gradient, weights))
			else:
				self.neurons[i].gradient = self.dev_hiperbolic(np.dot(gradient, weights))	
			
			self.weights[:] = self.learning_rate * self.neuron[i].gradient * inputs[:]	
		
class C_Layer:
		
	def __init__(self, n_maps, n_neurons, connections):

		self.maps = np.empty((n_maps), dtype = object)

		for i in xrange(0, n_maps):
			self.maps[i] = Map(n_neurons, connections)

		self.gradients = 0
		self.inputs = 0
		self.weights = 0

	def get_gradient(self, obj):

		self.gradients = np.zeros((obj.maps.size()))
		
class S_Layer:

	def __init__(self, n_maps, n_neurons, connections):

		self.maps = np.empty((n_maps), dtype = object)

		for i in xrange(0, n_maps):
			self.maps[i] = Map(n_neurons, connections)
	
class CNN:

	def __init__()
