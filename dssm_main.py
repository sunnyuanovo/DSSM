'''
Created on Jan 19, 2015

@author: lxh5147
'''
import theano
import theano.tensor as T
import os
import sys
import time
import numpy
#import theano.config

# Set lower precision float, otherwise the notebook will take too long to run
theano.config.floatX = 'float32'

import timeit
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

def get_minibatch(i, dataset_x, dataset_y):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_x = dataset_x[start_idx:end_idx]
    batch_y = dataset_y[start_idx:end_idx]
    return (batch_x, batch_y)


## early-stopping parameters tuned for 1-2 min runtime
def sgd_training(train_model, train_set_x, train_set_y, model_name='mlp_model',
                 # maximum number of epochs
                 n_epochs=2,
                 # look at this many examples regardless
                 patience=2,
                 # wait this much longer when a new best is found
                 patience_increase=2,
                 # a relative improvement of this much is considered significant
                 improvement_threshold=0.995,
                 batch_size=4):


    n_train_batches = train_set_x.shape[0] // batch_size

    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_x, minibatch_y = get_minibatch(minibatch_index, train_set_x, train_set_y)
            
            train_model(minibatch_x, minibatch_y)
    

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            minibatch_index += 1

    end_time = timeit.default_timer()

    print('The code ran for %d epochs, with %f epochs/sec (%.2fm total time)' %
          (epoch, 1. * epoch / (end_time - start_time), (end_time - start_time) / 60.))
    
    
    

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                )
                , dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        lin_output = T.dot(input, self.W)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W]

class CosineLayer(object):
    """

    Given two inputs [q1 q2 .... qmb] and [d1 d2 ... dmb],
    this class computes the pairwise cosine similarity 
    between positive pairs and neg pairs
    """
    def ComputeCosineBetweenTwoVectors(self, q_ind, d_ind, Q, D):
        # index is like (1,2)
        q = Q[q_ind]
        d = D[d_ind]
        qddot = T.dot(q,d)
        q_norm = T.sqrt((q**2).sum())
        d_norm = T.sqrt((d**2).sum())
        return qddot/(q_norm * d_norm)

    # for train, we need to compute a cosine matrix for (Q,D), then compute a final score
    def forward_train(self):   
        n_mbsize = self.Q.shape[0] # get mbsize as int, such as 1024
        
        # Next, we need to generate 2 lists of index
        # these 2 lists together have mbsize*(neg+1) element
        # after reshape, it should be (mbsize, neg+1) matrix
        index_Q = []
        index_D = []
        for tmp_index in range(n_mbsize):
            # for current sample, it's positive pair is itself
            index_Q += [tmp_index] * (self.n_neg+1)
            index_D += [tmp_index]
            index_D += [(tmp_index+self.n_shift+y)%n_mbsize for y in range(self.n_neg)]
        
        components, updates = theano.scan(ComputeCosineBetweenTwoVectors,
                                  outputs_info=None,
                                  sequences=[index_Q, index_D],
                                  non_sequences=[Q,D])
        
        components_reshape = T.reshape(components, (n_mbsize, self.n_neg+1))
        
        # for this matrix, each line is a prob distribution right now.
        components_reshape_softmax = T.nnet.softmax(components_reshape)
        
        # get the first column
        column1 = components_reshape_softmax[:,0]
        
        # get the final output
        self.output_train = - column1.sum()

    # for test, we only need to compute a cosine vector for (Q,D)
    def forward_test(self):   
        n_mbsize = self.Q.shape[0] # get mbsize as int, such as 1024
        
        # Next, we need to generate 2 lists of index
        # both are like (0, 1,   ..., 1023)
        index_Q = numpy.arange(n_mbsize)
        index_D = numpy.arange(n_mbsize)

        # components is a vector         
        components, updates = theano.scan(ComputeCosineBetweenTwoVectors,
                                  outputs_info=None,
                                  sequences=[index_Q, index_D],
                                  non_sequences=[Q,D])
        
        
        # get the final output
        self.output_test = components
        
    def __init__(self, Q, D, n_neg, n_shift):
        """ Initialize the parameters of the logistic regression

        :type inputQ and inputD: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        
        :type n_neg: int
        :param n_neg: number of negative samples
        
        :type n_shift: int
        :param n_out: shift of negative samples

        """
        # keep track of model input and target.
        # We store a flattened (vector) version of target as y, which is easier to handle
        self.Q = Q
        self.D = D
        self.n_neg = n_neg
        self.n_shift = n_shift
        
        self.forward_train()
        self.forward_train()
        
        # For this node, we don't have W or b as parameters
        # Therefore, we don't set params in this node
     

def load_data():
    # query w
    # query w'
    # for debug purpose, use random numbers
    rng = numpy.random.RandomState(1234)
    dataset = [theano.shared(numpy.asarray(
                    rng.uniform(
                        low=0.,
                        high=1.0,
                        size=(8, 2)
                    )
                ,config.floatX))
                  , theano.shared(numpy.asarray(
                    rng.uniform(
                        low=0.,
                        high=1.0,
                        size=(8, 2)
                    )
                ,config.floatX))  ]  
    return dataset

class MLP(object):
    def __init__(self, rng, input_Q, input_D, n_in, n_hidden, n_out, activation=T.tanh):
        self.input_Q = input_Q
        self.input_D = input_D

        # Build all necessary hidden layers and chain them
        # in current step, we just use one hidden layer, i.e. n_hidden = 1
        # first, build hidden layers for Q
        self.hidden_layers_Q = []
        layer_input = input_Q
        layer_n_in = n_in

        for nh in n_hidden:
            hidden_layer = HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=layer_n_in,
                n_out=nh,
                activation=activation)
            self.hidden_layers_Q.append(hidden_layer)

            # prepare variables for next layer
            layer_input = self.hidden_layer_Q.output # this is a matrix
            layer_n_in = nh
            
        # Next, build hidden layers for D
        self.hidden_layers_D = []
        layer_input = input_D
        layer_n_in = n_in

        for nh in n_hidden:
            hidden_layer = HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=layer_n_in,
                n_out=nh,
                activation=activation)
            self.hidden_layers_D.append(hidden_layer)

            # prepare variables for next layer
            layer_input = self.hidden_layer_D.output
            layer_n_in = nh
        

        self.cosine_layer = CosineLayer(
            Q = self.hidden_layers_Q[-1].output,
            D = self.hidden_layers_D[-1].output, 
            n_neg = 1, 
            n_shift = 1
            )

        
        # self.params has all the parameters of the model,
        # self.weights contains only the `W` variables.
        # We also give unique name to the parameters, this will be useful to save them.
        self.params = []
        self.weights = []
        layer_idx = 0
        for hl in self.hidden_layers_Q:
            self.params.extend(hl.params)
            self.weights.append(hl.W)
            for hlp in hl.params:
                prev_name = hlp.name
                hlp.name = 'layer_Q' + str(layer_idx) + '.' + prev_name
            layer_idx += 1
        
        layer_idx = 0
        for hl in self.hidden_layers_D:
            self.params.extend(hl.params)
            self.weights.append(hl.W)
            for hlp in hl.params:
                prev_name = hlp.name
                hlp.name = 'layer_D' + str(layer_idx) + '.' + prev_name
            layer_idx += 1

    # this returns a scalar of neg log likelihood
    def forward_train(self):
        return self.cosine_layer.output_train
    
    # this returns a vector of cosine values, for (Q,D) pairwise
    def forward_test(self):
        return self.cosine_layer.output_test

def nll_grad(mlp_model):
    loss = mlp_model.forward_train()
    params = mlp_model.params
    grads = theano.grad(loss, wrt=params)
    # Return (param, grad) pairs
    return zip(params, grads)

def sgd_updates(params_and_grads, learning_rate):
    return [(param, param - learning_rate * grad)
            for param, grad in params_and_grads]

def get_simple_training_fn(mlp_model, learning_rate):
    inputs = [mlp_model.input_Q, mlp_model.input_D]
    params_and_grads = nll_grad(mlp_model)
    updates = sgd_updates(params_and_grads, learning_rate=learning_rate)
    
    return theano.function(inputs=inputs, outputs=[], updates=updates)

def test_sim(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset=None, batch_size=10, n_hidden=500):
    """
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    if not  dataset: 
        dataset = load_data()

    train_set_x = dataset[0]
    train_set_y = dataset[1]
    
    
    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')
    # The labels coming from Fuel are in a "column" format
    y = T.matrix('y')

    n_in = 2
    n_out = 2

    mlp_model = MLP(
                    rng=rng,
                    input_Q=x,
                    input_D=y,
                    n_in=n_in,
                    n_hidden=[4],
                    n_out=n_out)

    lr = numpy.float32(0.1)

    train_model = get_simple_training_fn(mlp_model, lr)
    sgd_training(train_model, train_set_x, train_set_y)
    



if __name__ == '__main__':
    print "Git test\n"
#    test_sim()
