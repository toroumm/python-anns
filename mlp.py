"""

by jeferson de souza 16/08/2014

"""
"""

Slices Samples

a[start:end] # items start through end-1
a[start:]    # items start through the rest of the array
a[:end]      # items from the beginning through end-1
a[:]         # a copy of the whole array

a[-1]    # last item in the array
a[-2:]   # last two items in the array
a[:-2]   # everything except the last two items
a[:,1] #get the second column

slice(start, stop, increment)

"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pickle
import sys

'''

'''
class Neurons:

	def __init__(self	,conections):
		self.currentBias = random.random()
		self.weight = np.random.normal(0, 0.01,conections)
		self.previousBias = 0.0
		self.inputs = 0.0

	def __getitem__(self, Neuron):
		if isinstance(self.weight,slice):
			return self.__class__(self[x]
					     for x in xrange(*self.weight.indices.len(self)))	

'''

'''
class Layers(Neurons):

	def __init__(self,number_neurons,conections):

		self.neuron = np.empty((number_neurons),dtype = object)
		for i in range(0, int(number_neurons)):		
			self.neuron[i] = Neurons(conections)
		self.inputs = np.zeros((number_neurons))
		self.gradient = np.zeros((number_neurons))
		self.delta = np.zeros((number_neurons))
		self.n_neurons = number_neurons
		

	def __getitem__(self, Layer):
		if isinstance(self.neuron,slice):
			return self.__class__(self[x]
					     for x in xrange(*self.neuron.indices.len(self)))
	
class MLP(Layers):

	def __init__(self, *args):
		
		if(len(args)>=3):
			
			inputs = args[0] #numero de padroes de entrada
			outputs = args[1] # numero de neuronios na camada de saida
			hidden = args[2] # vetor que com dimensao do numero de camadas escondidas e os neuronios em cada uma
			
			if(len(args)==4):
				tag = args[3]
			else:
				tag = 'mlp'
			
			self.layer = np.empty((hidden.size+2), dtype = object)
			
			self.layer[0] = Layers(inputs,0)
			
			if(tag != 'dbn'):

				self.layer[1] = Layers(hidden[0],inputs)		 

				if hidden.size > 1:
					for i in range(2,self.layer.size-1):		
						self.layer[i] = Layers(hidden[i-1],hidden[i-2])

				self.layer[self.layer.size-1] = Layers(outputs,hidden[hidden.size-1])

			self.erro = 0.0
			self.quad_erro_train = []
			self.quad_erro_validation = []
			self.on_validation = False
			self.learningRate =  0.8
			self.momentum = 0.5
			self.learningDescent =  1
			self.epochs = 100
			self.plotar = 0
			self.max_normalization = 0
			self.min_normalization = 0

			self.max_normalization_out = 0
			self.min_normalization_out = 0

		else:
			print ('Invalid Arguments ')
			return 0

	def __getitem__(self, MLP):
		if isinstance(self.layer,slice):
			return self.__class__(self[z]
					     for x in xrange(*self.layer.indices.len(self))
					     	   for y in xrange(*self.layer.neuron.indices.len(self))
							 for z in xrange(*self.layer.neuron.weight.indices.len(self)))	

	def set_hidden_layer(self, _layer, index):

		self.layer[index] = Layers(len(_layer),self.layer[index-1].n_neurons)
		
		for i in xrange(0, len(_layer)):
			self.layer[index].neuron[i].weight = np.copy(_layer[i].weights)
			self.layer[index].neuron[i].currentBias = _layer[i].bias


	def set_learning_rate(self,value):
		self.learningRate = value

	def set_momentum(self,value):
		self.momentum = value

	def set_epochs(self,value):
		self.epochs = value

	def set_learning_descent(self,value):
		self.learningDescent = value

	def sigmoidal(self, vj):
		return 1/(1+np.exp(-vj))

	def __devSigmoidal(self,y):
		return y*(1.0-y)

	def get_error_train(self):
		return np.asarray(self.quad_erro_train)

	def get_error_validation(self):
		return np.asarray(self.quad_erro_validation)
		
	'''
		configure the samples and output set to train the MLP	
	'''

	def validation_set(self, inputs ,outputs):

		try: 
			a,b = outputs.shape
			self.validation_out = outputs
		except ValueError:
			self.validation_out = np.zeros((outputs.shape[0], 1))
			self.validation_out[:,0] = outputs[:]

		self.validation = inputs
		 
		self.on_validation = True
	'''
		this function is only test to verify if the mlp still works fine after update actions
		
		this test is available in the book "Data Mining Concepts and Techniques" pages 405, 406.
	'''	
		
	def teste(self):

		self.layer[1].neuron[0].weight[0] = 0.2
		self.layer[1].neuron[0].weight[1] = 0.4
		self.layer[1].neuron[0].weight[2] = -0.5
		self.layer[1].neuron[0].currentBias = -0.4
		
		self.layer[1].neuron[1].weight[0] = -0.3
		self.layer[1].neuron[1].weight[1] = 0.1
		self.layer[1].neuron[1].weight[2] = 0.2	
		self.layer[1].neuron[1].currentBias = 0.2
		
		self.layer[2].neuron[0].weight[0] = -0.3
		self.layer[2].neuron[0].weight[1] = -0.2
		self.layer[2].neuron[0].currentBias = 0.1

	'''
	variable: set_model [ 0 == training; 1 == validation_set (without backward); 2 == prediction mode] 

	forward function receive the set_model, samples and corresponding output, and carry out the first step of mlp
	'''
	 
	def __forward(self, set_model, _samples, _output = None):

		predict  = np.zeros((int(_samples.shape[0]),self.layer[self.layer.size-1].n_neurons))	

		try:
			_erro = np.zeros((_output.shape[0],_output.shape[1]))

		except Exception:
			_erro = np.zeros((_output.size,1))

		for ii in range(0,_samples.shape[0]):
			
			self.layer[0].inputs[:] = _samples[ii:ii+1]

			for i in range(0,self.layer[1].neuron.size):

				self.wx = self.layer[1].neuron[i].weight[:] * _samples[ii:ii+1]	
				h = self.wx
				self.layer[1].inputs[i] = self.sigmoidal(np.sum(self.wx[:]) + self.layer[1].neuron[i].currentBias)
			#print h, asd
			for i in range(2,self.layer.size):

				for j in range(0,self.layer[i].neuron.size):

					self.wx = self.layer[i].neuron[j].weight[:] * self.layer[i-1].inputs[:] 
							
					self.layer[i].inputs[j] = self.sigmoidal(np.sum(self.wx[:]) + self.layer[i].neuron[j].currentBias)
					
			for i in range(0, predict.shape[1]):
 		
				predict[ii,i] = self.layer[self.layer.size-1].inputs[i]

				_erro[ii,i] = _output[ii,i] - self.layer[self.layer.size-1].inputs[i]
				
			if(set_model == 0):
				self.__backward(_erro[ii])

		if(set_model == 0 or set_model == 1):
			return _erro

		if(set_model == 2):
			return predict
	'''
		backward function is call inside forward step the according set_model, if is situation training the weights are update 
	'''			
		
	def __backward(self,erro):

			self.layer[self.layer.size-1].gradient[:] = erro[:]*self.__devSigmoidal(self.layer[self.layer.size-1].inputs[:])	
						
			for i in range(self.layer.size-2,0,-1):
				
				self.sum = np.zeros((self.layer[i].neuron.size))
						
				for j in range(0, self.layer[i].neuron.size):
					
					for k in range(0, self.layer[i+1].neuron.size):

						self.sum[j] += self.layer[i+1].gradient[k]*self.layer[i+1].neuron[k].weight[j]
						
					self.layer[i].gradient[j] = self.__devSigmoidal(self.layer[i].inputs[j])*self.sum[j]
								
			
			for i in range(self.layer.size-2,-1,-1):
			
				for k in range(0, self.layer[i].neuron.size):
											
					for j in range(0, self.layer[i+1].neuron.size):
					
						self.layer[i+1].neuron[j].weight[k] += (self.momentum*self.layer[i+1].delta[j]) + self.learningRate * self.layer[i+1].gradient[j] * self.layer[i].inputs[k]
			
						
			for i in range(self.layer.size-1,0,-1):
			
				for j in range(0, self.layer[i].neuron.size):
					
					self.layer[i].neuron[j].currentBias += self.learningRate * self.layer[i].gradient[j]
					self.layer[i].delta[j] = self.learningRate * self.layer[i].gradient[j]
			
	'''
		calc square mean error (MSE)
	'''		
	def __square_error(self, erro):

		quadratic = 0
		for i in xrange(0, erro.shape[0]):
			for j in xrange(0, erro.shape[1]):
				quadratic += math.pow(erro[i,j],2)
		return quadratic / erro.size
	'''
		Normalize the data set
	'''	

	def __normalize(self, inputs, maxi, mini):
		inputs = inputs.astype(float)
		for i  in range(0, inputs.shape[1]):
		
			if((maxi[i]-mini[i])> 0):
				inputs[:,i] -= mini[i]
			
				inputs[:,i]  /= (maxi[i]-mini[i])
		return inputs

	'''
		Desnormalize the data set
	'''	

	def __denormalization(self, inputs,maxi,mini):
		
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] *= (maxi[i]+mini[i])
			inputs[:,i] += mini[i]
		return inputs

	'''
		Change the order of samples in data set
	'''
	def __shuffle(self, _samples, _output):

		try:
			a1,b1 = _output.shape
		except ValueError:
			asd = _output
			_output = np.zeros((asd.shape[0],1))
			_output[:,0] = asd[:]
			a1,b1 = _output.shape
	
		a2,b2 = _samples.shape			

		dados = np.concatenate((_samples,_output),axis = 1)
		dados = np.random.permutation(dados)
					
		_samples = dados[:,0:b2]
		
		_output = dados[:,b2:b2+b1]
		
		return _samples,_output

	'''
		set input data base and desire output
	'''
	
	def __training_set(self,inputs,outputs):	
		
		self.samples = inputs

		try: 
			a,b = outputs.shape
			self.out = outputs
		except ValueError:
			self.out= np.zeros((outputs.shape[0], 1))
			self.out[:,0] = outputs[:]
	
		self.max_normalization = np.zeros((inputs.shape[1]))
		self.min_normalization = np.zeros((inputs.shape[1]))

		self.max_normalization_out = np.zeros((self.out.shape[1]))
		self.min_normalization_out = np.zeros((self.out.shape[1]))
		percent = 1.1
		for i in xrange(0, self.max_normalization.shape[0]):
			self.max_normalization[i] = (np.max((inputs[:,i]),axis = 0))*percent
			self.min_normalization[i] = np.min((inputs[:,i]),axis = 0)
			if(self.min_normalization[i] > 0):
				self.min_normalization[i] = 0
		
		for i in xrange(0, self.max_normalization_out.shape[0]):
			self.max_normalization_out[i] = (np.max((self.out[:,i]),axis = 0))*percent
			self.min_normalization_out[i] = np.min((self.out[:,i]),axis = 0)
			if(self.min_normalization_out[i] > 0):
				self.min_normalization_out[i] = 0
				
		self.samples, self.out = self.__shuffle(self.samples, self.out )

	
	'''
		Getting started mlp train process 
	'''
		
	def train_mlp(self, inputs, outputs):

		self.__training_set(inputs, outputs)

		self.__train()

	'''
		keep training after the first train	
	'''

	def keep_training(self,epochs=100, inputs = 0,outputs = 0):

		if(inputs != 0 and outputs != 0):
			self.samples = self.__denormalization(self.samples)

			self.samples = np.concatenate(self.samples,inputs)
	
			self.out = self.__denormalization(self.out)

			self.out = np.concatenate(self.out,outputs)
		
			self.__training_set(inputs, outputs)

		self.set_epochs(epochs)

		self. __train()
		
	'''
		start the forward and backward pass
	'''	

	def __train(self):

		stop_train = 0.00
		status = 0

		t_in, t_out = self.__shuffle(self.samples, self.out)
				
		if(self.on_validation):
			v_in, v_out = self.__shuffle(self.validation, self.validation_out)

		for i in xrange(0, self.epochs):
			
			self.quad_erro_train.append(self.__square_error(self.__forward(0, t_in, t_out)))

			if(self.on_validation):
				self.quad_erro_validation.append(self.__square_error(self.__forward(1, v_in, v_out)))

			print 'Epoch', i, 'mse',self.quad_erro_train[i]
			
			if (stop_train is not 0.00 and stop_train < self.quad_erro_train[i]):
				status +=1
			else:
				status = 0
				
			if(status >= 10):
				break
				
			stop_train = self.quad_erro_train[i]

			self.learningRate *= self.learningDescent		
					
	'''
	
		return the predition of data
	'''
	def predict(self, inputs):

		inputs = self.__normalize(inputs,self.max_normalization, self.min_normalization)

		return self.__denormalization(self.__forward(2,inputs,inputs), self.max_normalization_out, self.min_normalization_out)

	'''
		show the learning curve
	'''

	def plot_learning_curve(self,path = 0,name = 'image.png',plot =1):
		
		plt.xlabel('Epochs')
		plt.ylabel('Quadratic Error')
		plt.title('Quadratic Error Curve')

		y = np.arange(0,len(self.quad_erro_train))		
		
		p2 = plt.plot(y,np.asarray(self.quad_erro_train))
			
		if(self.on_validation):
			p1 = plt.plot(y,np.asarray(self.quad_erro_validation))
			plt.legend([p1[0], p2[0]], ['Validation_set','Training_set'])	
		else:
			plt.legend([p2[0]],['Training_set'])

			plt.plot(y,np.asarray(self.quad_erro_train))
		if(path != 0):
			plt.savefig(path+'plot_'+name)
	
		if(plot == 1):
			plt.show()

		plt.clf()
		plt.cla()
		
	'''
	
		save net
	'''	

	def save_mlp(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp,save,pickle.HIGHEST_PROTOCOL)
		save.close()
		
		
	
	'''
	
		Load net (be carefull static method)
	'''	
	@staticmethod
	def load_mlp( path_and_namefile):

		#path_and_namefile += ".pk1"
		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)

			return mlp
		except:
			print ("Open file error, try again")

		

#******************************************************************************************************
#import numpy
#import sys
#sys.path.append(caminho)
#from mlp import MLP
'''
path = '/home/jeferson/Desktop/'

ann = MLP.load_mlp(path+'mlp.pk1')

ann.keep_training(100)

ann.plot_learning_curve()


inp = np.loadtxt(path+'iris.txt')

out = np.loadtxt(path+'iris_out.txt')

hide = np.array([1]) #numero de camadas

hide[0] = 10; #numero de neuronios em cada camada

ann = MLP(inp.shape[1], out.shape[1], hide)

ann.set_epochs(200)

ann.set_learning_rate(0.5)

ann.train_mlp(inp,out)

ann.save_mlp(ann, '/home/jeferson/Desktop/mlp')

ann.plot_learning_curve()

'''
