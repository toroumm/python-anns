import numpy as np
import matplotlib.pyplot as plt
import sys,os

'''
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
'''
path = '/home/jeferson/Dropbox/Public/Projects/include/teste_relu/'

path_wb = '/home/jeferson/Dropbox/Public/Projects/include/teste_relu/'

path_file = '/home/jeferson/Dropbox/experimento_fmri/dados/163/'

X = np.loadtxt(path_wb + 'input.txt')

y  = np.zeros((300),dtype=int)

y[100:200] = 1;
y[200:300] = 2;


'''
y  = np.zeros((163),dtype=int)

y[75:] = 1;
'''

# initialize parameters randomly

'''
classes = 3

h = 100 # size of hidden layer
W = 0.01 * np.random.randn(X.shape[1],h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,classes)
b2 = np.zeros((1,classes))

'''

'''
np.savetxt(path_wb+'relu_w1.txt',W.T)
np.savetxt(path_wb+'relu_b.txt',b)
np.savetxt(path_wb+'relu_w2',W2)
np.savetxt(path_wb+'relu_b2.txt',b2)
'''


W = np.loadtxt(path_wb+'relu_w1.txt').T
b = np.loadtxt(path_wb+'relu_b.txt').reshape(1,100)
W2 = np.loadtxt(path_wb+'relu_w2.txt').T
b2 = np.loadtxt(path_wb+'relu_b2.txt').reshape(1,3)

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
   	
  scores = np.dot(hidden_layer, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 50 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples


  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)

  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)

  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b

  dW = np.dot(X.T, dhidden)

  db = np.sum(dhidden, axis=0, keepdims=True)
 
  dW2 += reg * W2
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2
  
  hidden_layer = np.maximum(0, np.dot(X, W) + b)
  scores = np.dot(hidden_layer, W2) + b2
  predicted_class = np.argmax(scores, axis=1)
  if i % 50 == 0:
    print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

