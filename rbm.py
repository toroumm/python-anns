import numpy as np
import matplotlib.pyplot as plt
import random
import sys, os, subprocess
#from joblib import Parallel, delayed
import pickle



class Neuron:

	def __init__(self,connections):
		
		self.weights = 0
		if(connections > 0):
			self.weights = np.random.normal(0,0.01, connections)
			
		
		self.h_awk_prob = 0
		self.v_awk_prob = 0
		self.h_slp_prob = 0
		self.v_slp_prob = 0
		
		self.h_awk_status = 0
		self.h_slp_status = 0
		self.v_slp_status = 0
		
		self.bias = 0#random.random()
		self.momentum_bias = 0
		self.momentum = np.zeros((connections))
				
		
class RBM(Neuron):

	def __init__(self, _samples, n_neurons):

		self.min_normalization = 0
		self.max_nromalization = 0
	
		self.gaussian_var = np.zeros((_samples.shape[1]))

		self.gaussian_mean = np.zeros((_samples.shape[1]))

		#self.samples = self.gaussian_normalize(self.normalize(_samples))

		self.get_min_and_max(_samples)		

		self.samples = self.normalize(_samples)

		self.output = np.zeros((_samples.shape[0], _samples.shape[1]))
		
		self.validation = 0

		self.precision = 0
	
		self.input_layer = np.empty((_samples.shape[1]), dtype = object)

		self.hidden_layer = np.empty((n_neurons),dtype = object)

		self.learning_rate = 0.1

		self.alpha = 0.5
		
		self.gaussian_bernoulli = 1
		
		self.sigma = 0.01

		for i in xrange(0, self.input_layer.shape[0]):
			self.input_layer[i] = Neuron(0)

		for i in xrange(0, self.hidden_layer.shape[0]):
			self.hidden_layer[i] = Neuron(_samples.shape[1])
	@staticmethod
	def build_layer(path_neurons, path_bias):
		
		try:
			n = np.loadtxt(path_neurons)

			b = np.loadtxt(path_bias)
	
		except:
				
			print "Erro ao abrir arquivo"

			sys.exit()
		
		layer = np.empty((n.shape[0]),dtype = object)

		for i in xrange(0, layer.shape[0]):

			layer[i] = Neuron(0)

			layer[i].weights = np.copy(n[i])

			layer[i].bias = b[i]

		return layer


	def get_weights(self):

		layer = self.get_layer()		

		weight = np.zeros((layer.size,layer[0].weights.size))

		for i in xrange(0,weight.shape[0]):
			for j in xrange(0,weight.shape[1]):
				weight[i,j] = layer[i].weights[j]

		return weight
		

	def get_layer(self):

		return self.hidden_layer
		
	def set_sigma(self,value):
		self.sigma = value
		
	def set_gaussian_bernoulli(self, value): #boolean True or false
		self.gaussian_bernoulli = value
				
	def set_learning_rate(self,value):

		self.learning_rate = value
	
	def set_momentum(self,value):
		self.alpha = value	
		
	def set_validation(self,val):
	
		self.validation = self.normalize(val)

	def get_min_and_max(self, inputs):

		self.max_normalization = np.zeros((inputs.shape[1]))
		self.min_normalization = np.zeros((inputs.shape[1]))

		for i in xrange(0, self.max_normalization.shape[0]):
			self.max_normalization[i] =  np.max((inputs[:,i]),axis = 0)
			self.min_normalization[i] = np.min((inputs[:,i]),axis = 0)
			if(self.min_normalization[i] > 0):
				self.min_normalization[i] = 0.0
	
	def get_free_energy(self,samples):
		
		for i in xrange(0, samples.shape[0]):
			term_2 =0
			for j in xrange(0, self.hidden_layer.shape[0]):
				asd = 0
				
				for k in xrange(0, self.input_layer.shape[0]):
					
					asd += self.hidden_layer[j].weights[k] * samples[i,k]
				
				term_2 += np.log(1 + np.exp(self.sigmoidal(asd + self.hidden_layer[j].bias)))	
			term_1 = 0
			for j in xrange(0, self.input_layer.shape[0]):
				term_1 += samples[i,j] * self.input_layer[j].bias
				
		return (-1)*term_1 - term_2
				

			
	def normalize(self, inputs):
	
		inputs =inputs.astype(float)
		
		for i  in range(0, inputs.shape[1]):
			
			if((self.max_normalization[i]-self.min_normalization[i]) > 0):	
				inputs[:,i] -= self.min_normalization[i]		
				inputs[:,i] /=  (self.max_normalization[i]-self.min_normalization[i])
		 
		return np.random.permutation(inputs)

	def gaussian_desnormalize(self,inputs):

		for i in xrange(0,inputs.shape[0]):

			inputs[:,i] = (inputs[:,i] * self.gaussian_var[i]) + self.gaussian_mean[i] 

		return inputs

	def gaussian_normalize(self, inputs):

		for i in xrange(0,inputs.shape[0]):

			self.gaussian_var[i] = np.var(inputs[:,i])

			self.gaussian_mean[i] = np.mean(inputs[:,i])

			inputs[:,i] = (inputs[:,i] - np.mean(inputs[:,i])) / np.var(inputs[:,i])
			
		return inputs
			
	
	def keep_training(self, epochs=100, inputs = 0, _precision = 0.01, name_str_file =0, net_name = 'qualquer'):

		if(inputs != 0):
			
			self.samples = np.concatenate(self.samples,self.normalize(inputs))
	
		self.training_rbm(epochs, _precision,name_str_file, net_name)
		
	
	def denormalization(self, inputs):
		inputs = inputs.astype(float)
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] *= self.max_normalization[i]

		return inputs
	
	def save(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp,save,pickle.HIGHEST_PROTOCOL)
		
	@staticmethod
	def load( path_and_namefile):

		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)

			return mlp
		except:
			
			print ("Open file error, try again")

	def sigmoidal(self, vj):
		return 1/(1+np.exp(-vj))


	def training_rbm(self, epochs = 500, cds = 1,_precision = 0.001,name_str_file = 0, net_name = 'qualquer'):

		before = 0
	
		storage = []

		str_count = 0

		while(epochs >= 1):

			try:

				#self.run(self.samples)
				self.run_cds(self.samples,1,cds)

				res = np.copy(self.denormalization(self.samples))
	
				res = self.predict(res)
			
				diferenca = abs(before - self.precision)

				media = np.mean(abs(self.denormalization(self.samples) - res))

				#print "Epoch", epochs, 'Mean', media,'Precision',self.precision, 'Diference', diferenca
				
				a = self.get_free_energy(self.samples)
				b = self.get_free_energy(self.validation)
				
				print "Epoch", epochs, a, b, a-b,'Precision',self.precision
			
				before = self.precision

				values = [epochs,self.precision,diferenca,media ]

				storage.append(np.asarray(values))

				self.precision = 0
			
				if(name_str_file != 0):
					if(str_count >= 40):
						np.savetxt(name_str_file,np.asarray(storage))
						str_count = 0
						
				str_count +=1
	
				epochs-=1
			
				if(diferenca <= _precision):
					if(net_name != 0):
						self.save(self,net_name)
						
			except KeyboardInterrupt:
				if(net_name != 0):
					self.save(self,net_name)

				sys.exit()

	
	def update_weights(self,_samples):
	
		for j in xrange(0, self.hidden_layer.shape[0]):

			for k  in xrange(0, self.input_layer.shape[0]):
				
				
				self.hidden_layer[j].weights[k] += (self.alpha * self.hidden_layer[j].momentum[k]) + self.learning_rate*((self.hidden_layer[j].h_awk_status * _samples[k]) - (self.hidden_layer[j].h_slp_status * self.input_layer[k].v_slp_prob)) - (0.001 * self.hidden_layer[j].weights[k])
						
						
				self.hidden_layer[j].momentum[k] = ((self.hidden_layer[j].h_awk_status * _samples[k]) - (self.hidden_layer[j].h_slp_status * self.input_layer[k].v_slp_prob))
			
			self.hidden_layer[j].bias += (self.alpha * self.hidden_layer[j].momentum_bias) + self.learning_rate*(self.hidden_layer[j].h_awk_prob  - self.hidden_layer[j].h_slp_prob)
							
			self.hidden_layer[j].momentum_bias = (self.hidden_layer[j].h_awk_prob  - self.hidden_layer[j].h_slp_prob)		
					
		#Atualizacao bias entrada
		for j in xrange(0, self.input_layer.shape[0]):
			
			
			self.input_layer[j].bias += (self.input_layer[j].momentum_bias * self.alpha)+self.learning_rate*(self.input_layer[j].v_awk_prob - self.input_layer[j].v_slp_prob)
			
			self.input_layer[j].momentum_bias = (_samples[k]- self.input_layer[j].v_slp_prob)
	
	
	def get_gibbs_sampling(self,sampling):
	
		for i in xrange(sampling):
		
		#Calculo p(v' = 1 | h)
			for j in xrange(0, self.input_layer.shape[0]):

				self.input_layer[j].v_slp_prob  = 0
				for k in xrange(0, self.hidden_layer.shape[0]):
					self.input_layer[j].v_slp_prob += self.hidden_layer[k].h_slp_status * self.hidden_layer[k].weights[j]
					
				self.input_layer[j].v_slp_prob = self.sigmoidal(self.input_layer[j].v_slp_prob + self.input_layer[j].bias)
				
				if(self.gaussian_bernoulli == False):
					self.input_layer[j].v_slp_status = np.sum(np.random.binomial(1,self.input_layer[j].v_slp_prob,1)==1)
				
				else:
	       				self.input_layer[j].v_slp_prob /= self.hidden_layer.shape[0]
	       				
	       				self.input_layer[j].v_slp_prob += self.input_layer[j].bias
	       				
	       				self.input_layer[j].v_slp_status  = self.input_layer[j].v_slp_prob + (np.sum(np.random.normal(0,self.sigma,1))*self.sigma)
				
				
		#Calculo p(h' = 1 | v')
		
			for j in xrange(0, self.hidden_layer.shape[0]):

				self.hidden_layer[j].v_slp_prob = 0
				for k in xrange(0, self.input_layer.shape[0]):

					self.hidden_layer[j].h_slp_prob += self.hidden_layer[j].weights[k] * self.input_layer[k].v_slp_status
						
				self.hidden_layer[j].h_slp_prob = self.sigmoidal(self.hidden_layer[j].h_slp_prob + self.hidden_layer[j].bias)
			
				if(self.gaussian_bernoulli == False):
					self.hidden_layer[j].h_slp_status = np.sum(np.random.binomial(1,self.hidden_layer[j].h_slp_prob,1)==1)	
	       		
	       			else:
	       				self.hidden_layer[j].h_slp_prob /= self.input_layer.shape[0]
	       				
	       				self.hidden_layer[j].h_slp_prob += self.hidden_layer[j].bias
	       			      			
	 				self.hidden_layer[j].h_slp_status  = self.hidden_layer[j].h_slp_prob + (np.sum(np.random.normal(0,self.sigma,1))*self.sigma)
		
	
	def run_cds(self,_samples,model =1,n_samples=1):
	
		predict = np.zeros((_samples.shape[0],_samples.shape[1]))
		dbn = np.zeros((_samples.shape[0],len(self.hidden_layer)))
		
		self.precision = 0
		
		for i in xrange(0, _samples.shape[0]):
				
			#awake
			for j in xrange(0, self.hidden_layer.shape[0]):
				
				self.hidden_layer[j].h_awk_prob =  self.hidden_layer[j].h_slp_prob = self.sigmoidal(np.dot(self.hidden_layer[j].weights, _samples[i]) + self.hidden_layer[j].bias) 
				
				self.hidden_layer[j].h_awk_status = self.hidden_layer[j].h_slp_status = np.sum(np.random.binomial(1,self.hidden_layer[j].h_awk_prob,1)==1)
				
				self.get_gibbs_sampling(n_samples)
			
				dbn[i,j] = self.hidden_layer[j].h_slp_prob
				
			for j in xrange(0, self.input_layer.shape[0]):
			
				predict[i,j] = self.input_layer[j].v_slp_prob
				
				self.precision+= abs(_samples[i,j] - self.input_layer[j].v_slp_prob)
				
			
			#Atualizacao Pesos
			if(model == 1):
			
				self.update_weights(_samples[i])

		#retorna a saida da camada escondida	
		if(model == 0):
				
			return predict
		#retorna a reproducao da camada de entrada
		if(model == 2):
			return dbn

	def predict(self,inputs):
		
		return self.denormalization(self.run_cds(self.normalize(inputs),0,1))


	def predict_dbn(self,inputs):
	
		inputs = self.normalize(inputs)
		return self.run_cds(inputs,2,1)
		
		
	
	






