'''
by jeferson de souza 05/01/2015

'''

import cv2
import pickle
import sys
import numpy as np

from rbm import RBM
from mlp import MLP, Layers,Neurons

class DBN:

	def __init__(self, inputs, output, _hidden, _epochs):
		
		self.perceptron = MLP(inputs,output,_hidden, 'dbn')		

		self.hidden = _hidden
		self.epochs = _epochs

		self.samples = 0
		self.out = 0
		self.learning_rate = 0.6
		self.momentum = 0.5

	def set_learning_rate(self,value):
		self.learning_rate = value

	def set_momentum(self,value):
		self.momentum = value

	def keep_training(self,inputs=0,outputs= 0,epochs =100):

		self.perceptron.keep_training(inputs,outputs,epochs)

	def training_set(self, inputs, output):
	
		self.samples = inputs
		
		try:
			a = output.shape[1]		
			self.out = output

		except IndexError:

			self.out = np.zeros((output.shape[0],1))
			self.out[:,0] = output[:]

	def gaussian_normalize(inputs):

		for i in xrange(0,inputs.shape[1]):

			inputs[:,i] = (inputs[:,i] - np.mean(inputs[:,i])) / np.var(inputs[:,i])

		return inputs

	def gaussian_desnormalize(inputs):

		for i in xrange(0,inputs.shape[1]):

			inputs[:,i] = (inputs[:,i] - np.mean(inputs[:,i])) / np.var(inputs[:,i])

		return inputs

	def build_dbn(self, _inputs):

		for i in xrange(0, len(self.hidden)):	

			boltz = RBM(_inputs, self.hidden[i])

			boltz.training_rbm(self.epochs[i],0)
			
			self.perceptron.set_hidden_layer(boltz.get_layer(),i+1)

			_inputs = boltz.predict_dbn(_inputs)
			
			print 'Camada ', i, 'ok'
	
		self.perceptron.layer[len(self.perceptron.layer)-1] = Layers(self.out.shape[1], self.perceptron.layer[len(self.perceptron.layer)-2].n_neurons)
		
	
	def build_dbn_apart(self, file_weight, file_bias):

		for i in xrange(0, len(self.hidden)):

			layer = RBM.build_layer(file_weight[i], file_bias[i])
			print file_weight[i],  file_bias[i], layer[0].weights.size 
			#for j in xrange(0, layer.shape[0]):

			#	print layer[j].weights.size
			#sys.exit()
			self.perceptron.set_hidden_layer(RBM.build_layer(file_weight[i], file_bias[i]),i+1)

		self.perceptron.layer[len(self.perceptron.layer)-1] = Layers(self.out.shape[1], self.perceptron.layer[len(self.perceptron.layer)-2].n_neurons)
			
	def tuning_dbn(self):

		self.perceptron.set_learning_rate(self.learning_rate)

		self.perceptron.set_momentum(self.momentum)

		self.perceptron.set_epochs(self.epochs[len(self.epochs)-1])

		self.perceptron.train_mlp(self.samples,self.out)		

		#self.perceptron.save_mlp(self.perceptron, 'dbn_01')

		#self.perceptron.plot_learning_curve()

	def print_test(self):

		for ii in xrange(0,len(self.perceptron.layer)-1):
					
			for j in xrange(0,self.perceptron.layer[ii+1].n_neurons):

				print 'Layer', ii, 'Neuron', j, self.perceptron.layer[ii+1].neuron[j].weight
			
	def train_dbn(self, tag = 0, file_weight = 0, file_bias = 0):
		
		if(tag == 0):
			self.build_dbn((self.samples))
		elif(tag ==1):
			self.build_dbn_apart(file_weight,file_bias)

		for i in xrange(0, self.perceptron.layer.shape[0]):
			print 'camada ',i, ' neurons ',self.perceptron.layer[i].n_neurons

		
		self.tuning_dbn()

	def predict(self, inputs):

		return self.perceptron.predict(inputs)


	def save(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp,save,pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(path_and_namefile):

		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)

			return mlp
		except:
			print ("Open file error, try again")





