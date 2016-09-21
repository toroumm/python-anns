# coding=utf-8
'''
	by Jeferson de Souza

'''

import numpy as np
import pickle
import sys
import cv2
import matplotlib.pyplot as plt
import random
import os


class Neuron:
    def __init__(self, attributes):
        self.weight = np.random.normal(0, 0.1,attributes)
        self.erro = 0  # critetio para insercao de um no
        self.edge = []
        self.age = []
        self.delta_weight = 0
        self.distance = 0
        self.id = 0


class GNG(Neuron):
    def __init__(self, database):

        neuron_begin = 2
        self.mse_curve = []
        self.max_normalization = 0
        self.min_normalization = 0
        self.epochs = 500
        self.alpha = 0.005  # fator de criterio para insercao de um novo no entre os nos que cercam o valor
        self.beta = 0.0006  # fator de reducao dos pesos, quanto menor a variacao a rede tende a estabilizacao ai ela para (stop riteria)
        self.neighborhood = 2
        self.alpha_max = 80  # numero de iteracoes para retirar uma aresta
        self.space = database.shape[1]
        self.samples = database
        self.samples_img = np.array((1))
        self.lambd = 10  # numero de epocas para inserir um no
        # self.img = np.zeros((512,512,3), np.uint8)
        # self.img[::] = 255
        # self.video = 0

        self.neurons = []
        self.neurons_norm = 0

        for ii in xrange(0, neuron_begin):
            n = Neuron(self.space)
            n.id = ii
            self.neurons.append(n)
        neurons = self.neurons[:]
        for ii in xrange(0, neuron_begin):
            for i in xrange(0, neuron_begin):
                if (i != ii):
                    neurons[ii].edge.append(neurons[i].id)
                    neurons[ii].age.append(0)

    def sava_adjacent_matrix(self, path_file):
        x = np.zeros((len(self.neurons), len(self.neurons)))

        for i in xrange(0, len(self.neurons)):
            for j in xrange(0, len(self.neurons[i].edge)):
                if (i != j):
                    x[i, j] = self.alpha_max - self.neurons[i].age[j]
        np.savetxt(path_file, x)

    def get_epochs(self):
        return self.epochs

    def set_lambda(self, value):
        self.lambd = value

    def set_epochs(self, value):
        self.epochs = value

    def set_edge_age_max(self, value):
        self.alpha_max = value

    def set_alpha(self,value):
        self.alpha = value

    def set_beta(self,value):
        self.beta = value

    def winner(self, neurons):

        menor_1 = np.exp(10)
        menor_2 = np.exp(10)
        index_1 = 0
        index_2 = 0

        for i in xrange(0, neurons.size):

            if (neurons[i].distance < menor_1):
                index_1 = i
                menor_1 = neurons[i].distance
            if (neurons[i].distance > menor_1 and neurons[i].distance < menor_2):
                index_2 = i
                menor_2 = neurons[i].distance

        if (menor_1 > menor_2):
            aux = index_2
            index_2 = index_1
            index_1 = aux
        return index_1, index_2

    def remove_edges(self, neurons):

        for i in xrange(0, neurons.size):
            j = 0
            while (j < len(neurons[i].edge)):
                if neurons[i].age[j] >= self.alpha_max:

                    for z in xrange(0, neurons.size):
                        if (neurons[i].edge[j] == neurons[z].id):
                            neurons[z].age.pop(neurons[z].edge.index(neurons[i].id))
                            neurons[z].edge.pop(neurons[z].edge.index(neurons[i].id))

                            neurons[i].age.pop(neurons[i].edge.index(neurons[z].id))
                            neurons[i].edge.pop(neurons[i].edge.index(neurons[z].id))

                            j = 0
                            break
                if (len(neurons[i].edge) <= 0):
                    self.remove_nodes(neurons)
                    break
                j += 1

    def remove_nodes(self, neurons):
        for i in xrange(0, len(neurons)):
            try:
                if (len(neurons[i].edge) == 0):
                    self.neurons.pop(self.neurons.index(neurons[i]))
                    i = 0
            except Exception:

                print "Value", i
            # sys.exit()

    def insert_nodes(self):

        neuron = np.array(self.neurons[:])
        self.remove_nodes(neuron)

        bigger = 0
        index_1 = 0
        for i in xrange(0, neuron.size):
            if (neuron[i].erro > bigger):
                bigger = neuron[i].erro
                index_1 = i

        edges = np.array(neuron[index_1].edge)

        bigger = 0
        index_2 = 0

        for i in xrange(0, edges.size):
            for j in xrange(0, neuron.size):

                if (edges[i] == neuron[j].id):
                    if (neuron[j].erro > bigger or neuron[j].erro == 0 and index_2 == 0):
                        index_2 = j

        n = Neuron(self.space)
        n.edge.append(neuron[index_1].id)
        n.edge.append(neuron[index_2].id)
        n.age.append(0)
        n.age.append(0)
        n.erro = neuron[index_1].erro * self.alpha
        n.id = self.neighborhood
        n.weight[:] = (neuron[index_1].weight[:] + neuron[index_2].weight[:]) / 2
        try:
            neuron[index_1].age.pop(neuron[index_1].edge.index(neuron[index_2].id))
            neuron[index_1].edge.pop(neuron[index_1].edge.index(neuron[index_2].id))

            neuron[index_2].age.pop(neuron[index_2].edge.index(neuron[index_1].id))
            neuron[index_2].edge.pop(neuron[index_2].edge.index(neuron[index_1].id))

            neuron[index_1].edge.append(n.id)
            neuron[index_1].age.append(0)
            neuron[index_1].erro *= self.alpha

            neuron[index_2].edge.append(n.id)
            neuron[index_2].age.append(0)
            neuron[index_2].erro *= self.alpha
        except Exception:
            print neuron[index_1].age, neuron[index_1].edge
            print neuron[index_2].age, neuron[index_2].edge
            sys.exit()
        self.neurons.append(n)
        self.neighborhood += 1

    def get_map_database(self, database):

        hist = np.zeros((database.shape[0]))

        for ii in xrange(0, database.shape[0]):

            neuron = np.array(self.neurons[:])

            for i in xrange(0, len(self.neurons)):
                neuron[i].distance = np.sqrt(np.sum(np.power(neuron[i].weight[:] - database[ii,], 2)))

            first, second = self.winner(neuron)

            hist[ii] = first

        return hist

    def get_map_gng(self, database, out=np.array([0])):

        if (out.shape[0] == 1):
            hist = np.zeros((len(self.neurons), 1))
        else:
            hist = np.zeros((len(self.neurons),1+ np.unique(out).shape[0]))

        database = self.normalize(database)
        # out =  vetor de classe a partir de 1 até n classes
        for ii in xrange(0, database.shape[0]):

            neuron = np.array(self.neurons[:])

            for i in xrange(0, len(self.neurons)):
                neuron[i].distance = np.sqrt(np.sum(np.power(neuron[i].weight[:] - database[ii,], 2)))

            first, second = self.winner(neuron)

            hist[first, 0] += 1

            if (out.shape[0] != 1):

                hist[first, int(out[ii])] += 1
            # print ii,first,second

        return hist

    def run(self):
        count_video = 0
        for ii in xrange(0, self.samples.shape[0]):

            neuron = np.array(self.neurons[:])

            for i in xrange(0, len(self.neurons)):
                neuron[i].distance = np.sqrt(np.sum(np.power(neuron[i].weight[:] - self.samples[ii,], 2)))

            first, second = self.winner(neuron)

            neuron[first].erro += neuron[first].distance

            neuron[first].delta_weight = np.sum(self.alpha * abs(self.samples[ii,] - neuron[first].weight[:]))

            neuron[first].weight[:] += self.alpha * (self.samples[ii,] - neuron[first].weight[:])

            for i in xrange(len(neuron[first].age)):
                neuron[first].age[i] += 1

            edges = np.array(neuron[first].edge)

            for i in xrange(0, edges.size):
                for j in xrange(0, neuron.size):
                    if (edges[i] == neuron[j].id):
                        neuron[j].weight[:] += self.beta * (self.samples[ii,] - neuron[j].weight[:])

            if (neuron[second].id in neuron[first].edge):

                neuron[first].age[neuron[first].edge.index(neuron[second].id)] = 0
                neuron[second].age[neuron[second].edge.index(neuron[first].id)] = 0

            else:
                neuron[first].edge.append(neuron[second].id)
                neuron[first].age.append(0)

                neuron[second].edge.append(neuron[first].id)
                neuron[second].age.append(0)

            self.remove_edges(neuron)

            self.remove_nodes(neuron)

            for i in xrange(0, neuron.size):
                neuron[i].erro *= 0.001

    def normalize_neurons(self, width, height, attr_1, attr_2):

        norm = np.zeros((len(self.neurons), self.samples.shape[1]))

        truco = np.array(self.neurons)

        max_1 = np.max(self.samples[:, attr_1])
        max_2 = np.max(self.samples[:, attr_2])
        min_1 = np.min(self.samples[:, attr_1])
        min_2 = np.min(self.samples[:, attr_2])

        for i in xrange(0, norm.shape[0]):
            norm[i, attr_1] = width * ((max_1 - truco[i].weight[attr_1]) / (max_1 - min_1))
            norm[i, attr_2] = height * ((max_2 - truco[i].weight[attr_2]) / (max_2 - min_2))

        return norm

    def normalize_img(self, width, height, attr_1, attr_2):

        self.samples_img = np.copy(self.samples)

        max_1 = np.max(self.samples_img[:, attr_1])
        max_2 = np.max(self.samples_img[:, attr_2])
        min_1 = np.min(self.samples_img[:, attr_1])
        min_2 = np.min(self.samples_img[:, attr_2])

        self.samples_img[:, attr_1] = width * ((max_1 - self.samples_img[:, attr_1]) / (max_1 - min_1))
        self.samples_img[:, attr_2] = height * ((max_2 - self.samples_img[:, attr_2]) / (max_2 - min_2))

    def training_gng(self, epochs=1000):
        lambda_counter = 0
        stop_criteria = 0
        old_total = 0

        # self.record_init()

        # for i in xrange(0,epochs):
        i = 0
        while (1 and epochs > i):
            self.run()
            lambda_counter += 1
            if (lambda_counter >= self.lambd):
                lambda_counter = 0
                self.insert_nodes()
            soma = 0
            neurons = np.array(self.neurons)
            for j in xrange(0, neurons.size):
                soma += neurons[j].delta_weight
            total = soma / neurons.size
            print "epoca", i, "lambda ", lambda_counter, "Neurons ", len(
                self.neurons), " MSE", total, "Parada", stop_criteria

            self.mse_curve.append(total)

            if (abs(total - old_total) < 0.000001):
                stop_criteria += 1
            else:
                if (stop_criteria >= 1):
                    stop_criteria -= 1

            old_total = total

            if (stop_criteria == 20):
                break
            i += 1

    def get_weights(self):

        neurons = np.array(self.neurons)

        weight = np.zeros((neurons.size, self.space))

        print "Memoria ", hex(id(weight))

        for i in xrange(0, weight.shape[0]):
            for j in xrange(0, weight.shape[1]):
                weight[i, j] = neurons[i].weight[j]
        return weight

    def training_set(self, inputs):

        self.max_normalization = np.zeros((inputs.shape[1]))
        self.min_normalization = np.zeros((inputs.shape[1]))

        for i in xrange(0, self.max_normalization.shape[0]):
            self.max_normalization[i] = np.max((inputs[:, i]), axis=0)
            self.min_normalization[i] = np.min((inputs[:, i]), axis=0)
            if (self.min_normalization[i] > 0):
                self.min_normalization[i] = 0

        self.samples = self.normalize(inputs)

    def normalize(self, inputs):
        '''
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] -= self.min_normalization[i]
			inputs[:,i]  /= (self.max_normalization[i]-self.min_normalization[i])
		'''
        return inputs

    def save_gng(self, mlp, path_and_namefile):

        self.video = 0

        path_and_namefile += ".pk1"

        with open(path_and_namefile, 'wb') as save:
            pickle.dump(mlp, save, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_gng(path_and_namefile):

        try:
            with open(path_and_namefile, 'rb') as input:
                mlp = pickle.load(input)

            return mlp
        except:
            print ("Open file error, try again")

    def plot_learning_curve(self):

        plt.xlabel('Epochs')
        plt.ylabel('Delta Change')
        plt.title('Graphic')

        y = np.arange(0, len(self.mse_curve))

        p2 = plt.plot(y, np.array(self.mse_curve))

        plt.legend([p2[0]], ['Training_set'])

        plt.show()


def analise_gng(dataset, n_vezes, path):

    for i in xrange(n_vezes):

        g = GNG(dataset)

        g.set_edge_age_max(50)

     #   g.set_alpha(0.0005)

        g.training_gng(1000)

        g.save_gng(g, path + 'rede_' + str(i))

        g.sava_adjacent_matrix(path + 'mat_adj_'+str(i))

        np.savetxt(path + 'map_database_'+str(i), g.get_map_database(dataset))

        np.savetxt(path + 'map_gng_'+str(i), g.get_map_gng(dataset, out))


path = '/home/jeferson/Dropbox/experimento_fmri/dados/163/'

dataset = np.loadtxt(path + 'frmi_input_geral_a4.txt')

out = np.loadtxt(path + 'output_163.txt')

out[out[:, 0] == 1] = 2
out[out[:, 0] == 0] = 1
out = out[:, 0]

path = '/home/jeferson/Dropbox/deep_learning/capitulos_mestrado/capitulos/resultados/'

analise_gng(dataset,20,path+'analise_gng/04/')

sys.exit(0)

'''
for i in xrange(2,8):

    g = GNG.load_gng(path+'base'+str(i)+'/rede_02.pk1')

    print g.alpha_max, g.epochs

sys.exit()
'''


g = GNG(dataset)


g.set_edge_age_max(20)

g.set

g.training_gng(300)

g.save_gng(g, path + 'rede_02')

g.sava_adjacent_matrix(path + 'mat_base_04')

np.savetxt(path + 'map_database', g.get_map_database(dataset))

np.savetxt(path + 'map_gng', g.get_map_gng(dataset, out))

path_files = '/home/jeferson/Dropbox/folds_fmri_cocaina/'

arquivo = 'base_4_fmri_teste_fold'

arquivo_out = 'base_4_fmri_teste_out_fold'


for i in xrange(0,10):

    x = np.loadtxt(path_files+arquivo+str(i+1))

    y = np.loadtxt(path_files + arquivo_out + str(i + 1))

    y[y[:,0]==1] =2

    y[y[:,0]==0] =1

    y = y[:,0]

    np.savetxt(path + 'mat_base_4_hist_'+str(i+1), g.get_map_gng(x,y))

