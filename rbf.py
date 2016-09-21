import numpy as np


class neuron:

	def __init__(self, conections, centroides):

		self.weights = np.random.random((conections))
		self.centers = np.zeros((centroides))

class rbf(neuron): 

	def __init__(self,input_neuron, hidden_neurons, output_neurons):

		self.hidden_layer = np.empty((hidden_neurons), dtype = object)

		for i in xrange(0, hidden_neurons):

			self.hidden_layer[i] = neuron(output_neurons, input_neurons)

		self.output_layer = np.zeros((output_neurons))

		self.epochs = 10

		self.learning_rate = 0.3

		self.centers = np.zeros((inputs_neuron))

		self.neighbors = 5

	def training_set(self,inputs, outputs, number_class):

		self.input  = inputs
		self.output = outputs
		self.n_class = number_class

	def define_centroide(self):
		
		mat_out = self.output

		for i in xrange(0, self.input.shape[1]):
			center = 0.0
			for j in xrange(0, self.input.shape[0]):
							
				self.centers[i] += self.input[i,j]

			self.centers[i] /= self.input.shape[0]
			
	def define_neighbors(self):

			

	
	def run_training(self):

		
