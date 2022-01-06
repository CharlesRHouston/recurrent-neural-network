"""
Implementation:
- tanh activation functions
- softmax output function
- categorical cross-entropy loss
- Xavier initialization
- Adam optimizer
- mini-batch gradient descent
- gradient clipping

Dimensions:
x:   (n_x, m, T_x)
a:   (n_a, m, T_x)
y:   (n_y, m, T_y)
Wax: (n_a, n_x)
Waa: (n_a, n_a)
Wya: (n_y, n_a)
ba:  (n_a, 1)
by:  (n_y, 1)

Class attributes:
n_a - dimension of hidden units

Class optimize method:
data - list of batches e.g. [[x1, y1], [x2, y2], ...]
        - xi: three dimensional numpy array (n_x, m, T_x)
        - yi: three dimensional numpy array (n_y, m, T_y)
epochs - number of times to pass through data; default = 20
learn - learning rate, alpha; default = 0.01
max_grad - gradient clipping value; default = 10

References:
https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1 
deeplearning.ai specialisation: sequence models
"""

# modules

import numpy as np

# recurrent class

class RecurrentNeuralNetwork:
    def __init__(self, n_a):
        self.n_a = n_a
    
    # activation function

    def act1(self, z): # tanh
        return np.tanh(z)
    
    def act1_(self, z): # derivative of act1
        return 1 - np.tanh(z)**2
    
    # output function
    
    def act2(self, z): # softmax
        for i in range(z.shape[1]):
            z[:,i] = np.exp(z[:,i])/np.sum(np.exp(z[:,i]))
        return z
    
    # loss function
    
    def loss(self, y_hat, y, eps = 1e-8):
        return -y*np.log(y_hat + eps)
    
    def loss_(self, y_hat, y): # derivative of both loss and act2
        return y_hat - y
    
    # Xavier initialisation

    def initialise_weights(self, n_x, n_y):
        params = {}
        params["Waa"] = np.random.randn(self.n_a, self.n_a)*np.sqrt(1/self.n_a)
        params["Wax"] = np.random.randn(self.n_a, n_x)*np.sqrt(1/n_x)
        params["Wya"] = np.random.randn(n_y, self.n_a)*np.sqrt(1/self.n_a)
        params["ba"]  = np.ones((self.n_a, 1))
        params["by"]  = np.ones((n_y, 1))
        return params
    
    # forward propagation through cell

    def rnn_cell_forward(self, params, x_t, a_prev):
        # extract parameters
        Waa, Wax, Wya, ba, by = params["Waa"], params["Wax"], params["Wya"], params["ba"], params["by"]
        # forward propagate through cell
        z_t = np.dot(Waa, a_prev) + np.dot(Wax, x_t) + ba       # (n_a, m)
        a_t = self.act1(z_t)                                    # (n_a, m)
        q_t = np.dot(Wya, a_t) + by                             # (n_y, m)
        y_t = self.act2(q_t)                                    # (n_y, m)
        return z_t, a_t, y_t
    
    # forward propagation
    
    def rnn_forward(self, X, Y, params):
        # dimensions
        n_a = params["Waa"].shape[0]
        n_x, m, T_x = X.shape
        n_y, m, T_y = Y.shape
        # initialise parameters
        Z = np.zeros((n_a, m, T_x))
        A = np.zeros((n_a, m, T_x + 1))
        Y_hat = np.zeros((n_y, m, T_y))
        # initialise hidden state
        a_prev = np.zeros((n_a, m))
        A[:,:,0] = a_prev
        # forward propagation
        for t in range(T_x):
            x_t = X[:,:,t]
            z_t, a_prev, y_t = self.rnn_cell_forward(params, x_t, a_prev)
            # store results
            Z[:,:,t] = z_t
            A[:,:,t+1] = a_prev
            Y_hat[:,:,t] = y_t
        cache = [Z, A]
        error = np.sum(self.loss(Y_hat, Y))/m
        return Y_hat, cache, error
    
    # backpropagation

    def rnn_backward(self, X, Y, params, Y_hat, cache):
        # dimensions
        n_a = params["Waa"].shape[0]
        n_x, m, T_x = X.shape
        n_y, m, T_y = Y.shape
        # required parameters
        Waa = params["Waa"]
        Wya = params["Wya"]
        # cached values
        Z, A = cache
        # initialize gradient storage
        gradients = {}
        gradients["dWaa"] = np.zeros((n_a, n_a))
        gradients["dWax"] = np.zeros((n_a, n_x))
        gradients["dWya"] = np.zeros((n_y, n_a))
        gradients["dba"]  = np.zeros((n_a, 1))
        gradients["dby"]  = np.zeros((n_y, 1))
        # iterate backwards through timesteps
        for t1 in range(T_x-1, -1, -1):
            Y_hat_t = Y_hat[:,:,t1]
            Y_t = Y[:,:,t1]
            A_t = A[:,:,t1+1]
            Z_t = Z[:,:,t1]
            # gradients for output layer parameters
            dWya = np.dot(self.loss_(Y_hat_t, Y_t), A_t.T)                       # (n_y, n_a)
            dby  = np.sum(self.loss_(Y_hat_t, Y_t), axis = 1, keepdims = True)   # (n_y, 1)
            # initialize gradients for recurrent parameters
            dWaa = np.zeros((n_a, n_a))
            dWax = np.zeros((n_a, n_x))
            dba  = np.zeros((n_a, 1))
            # backpropagation through each prior timestep
            for t2 in range(t1, -1, -1):
                # working gradient
                if t2 == t1:
                    dz = np.dot(Wya.T, self.loss_(Y_hat_t, Y_t))*self.act1_(Z_t)  # (n_a, m)
                else:
                    dz = np.dot(Waa.T, dz)*self.act1_(Z[:,:,t2])             # (n_a, m)
                # calculate gradients at timestep
                dWaa_t = np.dot(dz, A[:,:,t2].T)                # (n_a, n_a)
                dWax_t = np.dot(dz, X[:,:,t2].T)                # (n_a, n_x)
                dba_t  = np.sum(dz, axis = 1, keepdims = True)  # (n_a, 1)
                # update overall gradient
                dWaa += dWaa_t
                dWax += dWax_t
                dba  += dba_t
            # update weights
            gradients["dWaa"] += dWaa
            gradients["dWax"] += dWax
            gradients["dWya"] += dWya
            gradients["dba"]  += dba
            gradients["dby"]  += dby
        # divide by batch size
        for key in gradients.keys():
            gradients[str(key)] /= m
        return gradients
    
    # gradient clipping
    
    def clip(self, gradients, max_grad):
        dWaa, dWax, dWya, dba, dby = gradients["dWaa"], gradients["dWax"], gradients["dWya"], gradients["dba"], gradients["dby"]
        for gradient in [dWaa, dWax, dWya, dba, dby]:
            gradient = np.clip(gradient, -max_grad, max_grad, out = gradient)
        return {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
    
    # initialise Adam

    def initialise_adam(self, params):
        V = {}
        S = {}
        for key in params.keys():
            V["d" + str(key)] = np.zeros_like(params[str(key)])
            S["d" + str(key)] = np.zeros_like(params[str(key)])
        return V, S
    
    # update weights and Adam

    def update_adam(self, params, gradients, V, S, learn, t, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        V_correct = {}
        S_correct = {}
        for key in gradients.keys():
            # momentum
            V[str(key)] = beta1*V[str(key)] + (1-beta1)*gradients[str(key)]
            V_correct[str(key)] = V[str(key)]/(1-beta1**t)
            # RMSprop
            S[str(key)] = (beta2*S[str(key)] + (1-beta2)*gradients[str(key)]**2)
            S_correct[str(key)] = S[str(key)]/(1-beta2**t)
        # update parameters
        for key in params.keys():
            params[str(key)] -= learn*V_correct["d" + str(key)]/(np.sqrt(S_correct["d" + str(key)]) + eps)
        return params, V, S
    
    # Adam optimization

    def optimize(self, data, epochs = 20, learn = 0.01, max_grad = 10):
        # storage of loss for each iteration
        loss_per_epoch = np.zeros(epochs)
        # shapes
        n_x = data[0][0].shape[0]
        n_y = data[0][1].shape[0]
        # generate parameters
        params = self.initialise_weights(n_x, n_y)
        # initialise adam
        V, S = self.initialise_adam(params)
        t = 1
        # training
        for epoch in range(epochs):
            for batch in data:
                # extract
                X, Y = batch
                # forward propagation
                Y_hat, cache, error = self.rnn_forward(X, Y, params)
                # backward propagation
                gradients = self.rnn_backward(X, Y, params, Y_hat, cache)
                gradients = self.clip(gradients, max_grad)
                # adam
                params, V, S = self.update_adam(params, gradients, V, S, learn, t)
                t += 1
                # print(round(error, 3))
            # store loss
            loss_per_epoch[epoch] = error
            print(f"epoch: {epoch+1}")
            print(f"loss: {round(error, 3)}\n")
        
        return loss_per_epoch, params
    
    
    