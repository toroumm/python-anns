import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

class PCA:
	
	@staticmethod
	def calc(inputs, variance = 0.95):
		
		asd = ((inputs - np.mean(inputs.T,axis = 1))/np.std(inputs.T, axis = 1)).T
		
		eigen_values, eigen_vectors = np.linalg.eig(np.cov(asd))

		scores = np.dot(eigen_vectors.T,asd).T

		soma = 0
		index = 0
		for i in xrange(0,eigen_values.shape[0]):
			if(soma < variance):
				soma += eigen_values[i] / np.sum(eigen_values)
				index += 1
		
		return eigen_values, eigen_vectors , scores[:,:index]

class Neuron:

	def __init__(self, connections, side_connections):
		
		self.weight = np.random.random((connections))
		self.side_weight = np.random.random((side_connections))		
		self.output = 0
		self.late_weight = 0
		self.change = 0
		self.update = 0 

class Ann_PCA (Neuron):

	def __init__(self, in_inputs, out_neuron):

		self.max_normalization = 0
		self.min_normalization = 0
		self.learning_rate = random.random()
		self.mu = random.random()
		self.epochs = 300
		self.stop_criteria = 1
		self.erro = []
		self.precision = pow(10,-4)
		
		
		
		self.layer = np.empty((out_neuron), dtype = object)

		for i in xrange(0, out_neuron):
			self.layer[i] = Neuron(in_inputs,i)

	def set_precision(self,value):
		self.precision = value

	def set_learning_rate(self, value):
		self.learning_rate = value

	def get_pesos(self):
		return self.pesos

	def set_mu_rate(self,value):
		self.mu  = value
		
	def set_epochs(self, value):
		self.epochs = value

	def normalize(self, inputs):
		
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] -= self.min_normalization[i]
			inputs[:,i]  /= (self.max_normalization[i]-self.min_normalization[i])

		return inputs
		
	def training_set(self,inputs):

		self.max_normalization = np.zeros((inputs.shape[1]))
		self.min_normalization = np.zeros((inputs.shape[1]))

		for i in xrange(0, self.max_normalization.shape[0]):

			self.max_normalization[i] = np.max((inputs[:,i]),axis = 0)
			self.min_normalization[i] = np.min((inputs[:,i]),axis = 0)

			if(self.min_normalization[i] > 0):
				self.min_normalization[i] = 0

		self.samples = self.normalize(inputs)	

		#self.samples = ((inputs - np.mean(inputs.T,axis = 1))/np.std(inputs.T, axis = 1))	

	def get_eigenvector(self,n):
		return self.layer[n].weight

	def normalize(self, inputs):
		
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] -= self.min_normalization[i]
			inputs[:,i]  /= (self.max_normalization[i]-self.min_normalization[i])
		return inputs


	def forward(self, _samples, prediction):

		output = np.zeros((_samples.shape[0] , self.layer.shape[0]))
		
		for ii in xrange(0, _samples.shape[0]):

			for i in xrange(0, self.layer.shape[0]):

				soma = 0
				for j in xrange(0, i):
					soma += self.layer[j].output * self.layer[i].side_weight[j]

				self.layer[i].output = (np.sum(self.layer[i].weight[:] * _samples[ii]) + soma)

				output[ii,i] = self.layer[i].output
	
			if(prediction ==0):
				self.updating_side(_samples[ii])   		
				
		if(prediction == 1):
			return output


	def updating_side(self,_samples):
		
		x = 0
		for i in xrange(0, self.layer.shape[0]):
			if(self.layer[i].update == 1):
				x = i

		for j in xrange(0, self.layer[x].weight.shape[0]):

			self.layer[x].weight[j] += self.learning_rate *(( _samples[j] * self.layer[x].output) -  (pow(self.layer[x].output,2) * self.layer[x].weight[j]))
			print self.layer[x].weight[j]

		for j in xrange(0, self.layer[x].side_weight.shape[0]):
			self.layer[x].side_weight[j] += self.mu *(-1) * self.layer[x].output * self.layer[j].output
			

	def plot_convergence_curve(self):

		plt.xlabel('y')
		plt.ylabel('x')
		plt.title('Convergence Curve')
		
		a = np.array(self.erro[0])
		b = np.array(self.erro[1])

		plt.plot(a[:b[0]])
		for i in xrange(1, self.layer.shape[0]-1):
			asd = np.zeros((b[i-1]))
			c = np.concatenate((asd,a[b[i-1]:b[i]]))
			plt.plot(c)		

		plt.show()

	def normalize_weigths(self):

		for i in xrange(0, self.layer.shape[0]):

			modulo = np.linalg.norm(self.layer[i].weight)
			
			for j in xrange(0, self.layer[i].weight.shape[0]):
				self.layer[i].weight[j] /= modulo

						
	def training_pca(self):
		
		x = 0
		i = 0
	
		lista = []
		posicao = np.zeros((self.layer.shape[0]-1))
		while True:
			try:
				self.layer[x].late_weight = np.copy(self.layer[x].weight)

				self.forward(self.samples, 0)
			
				k = np.sqrt(np.sum(np.power(self.layer[x].late_weight - self.layer[x].weight,2)))

				lista.append(k)
				#abs(np.linalg.norm(self.layer[x].weight) - 1.0) <= pow(10,-4) and 
				if(k <= self.precision):
					self.layer[x].change += 1
					
				else:
					self.layer[x].change = 0
					

				if(self.layer[x].change > 300) :
					self.layer[x].update = 0
					
					if(x < self.layer.shape[0]-2):
						posicao[x] = i						
						x +=1
						self.layer[x].update = 1
					else:
						break
			
				print i,x,self.layer[x].change, np.linalg.norm(self.layer[x].weight),k
				i += 1

			except	KeyboardInterrupt:
				posicao[x] = i	
				x +=1
				if(x >= self.layer.shape[0]-1):
					break
		self.erro.append(lista)
		self.erro.append(posicao)		

	def save_pca(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp,save,pickle.HIGHEST_PROTOCOL)
		
	@staticmethod
	def load_pca( path_and_namefile):

		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)

			return mlp
		except:
			print ("Open file error, try again")

	def predict(self, inputs):
			
		return self.forward(inputs, 1)

'''
		
database = np.loadtxt("/home/jeferson/Desktop/uci_data_sets/wine.txt")

ann = Ann_PCA.load_pca("/home/jeferson/Desktop/Play_Py/pca_wine.pk1")#Ann_PCA(database.shape[1],database.shape[1])

#ann.training_set(database)

#ann.set_learning_rate(0.0001)

#ann.set_mu_rate(0.0001)

#ann.training_pca()

output = ann.predict(database)

a_value, a_vector, score = PCA.calc(database,1)

print np.var(output, axis = 0)

#ann.plot_convergence_curve()

#ann.save_pca(ann, "pca_twiter")


print "ANN "
for i in xrange(0, ann.layer.shape[0]):
	print ann.layer[i].weight

print "PCA ", a_vector.T

print "ANN ", np.mean(output[:])

a_value, a_vet, score = PCA.calc(database,0.95)

print "tam ",score.shape, 1.0 - (score.shape[1] / database.shape[1]),database.shape[1]

print 1- (float(score.shape[1]) /  float(database.shape[1])),database.shape[1],score.shape[1] 


print "variancia ", float(a_value[0] + a_value[1]) / float(np.sum(a_value[:])), 


ann = Ann_PCA.load_pca("/home/jeferson/Desktop/Play_Py/pca_breast_cancer.pk1")

print ann.erro[1]
plt.xlabel('First Componente')
plt.ylabel('Second Componente')
plt.title('Two First Eigen Values (Mamographic Masses Data Base)')

p1, = plt.plot(score[:14,0],score[:14,1], '^g')
p2, = plt.plot(score[15:49,0],score[15:49,1],'^r')	
p3, = plt.plot(score[50:603,0],score[50:603,1],'+k')
p4, = plt.plot(score[604:949,0],score[604:949,1],'bo')
p5, = plt.plot(score[949:,0],score[949:,1],'go')

plt.legend([p1,p2,p3,p4,p5], ['Class 1','Class 2','Class 3','Class 4','Class 5'])

plt.show()
'''

