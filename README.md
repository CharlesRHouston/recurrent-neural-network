# Recurrent Neural Network (From Scratch)

### Description
Recurrent neural network class coded up from scratch in python. No external frameworks such as Tensorflow or PyTorch were used. The class is contained in `recurrent_net.py`, while experiments are performed in `predict_consonant_vowel.py` and `five_letter_language_model.py`.

### Documentation

Implementation:
* tanh activation functions
* softmax output function
* categorical cross-entropy loss
* Xavier initialization
* Adam optimizer
* mini-batch gradient descent
* gradient clipping

Dimensions:
* **x**:   (n_x, m, T_x)
* **a**:   (n_a, m, T_x)
* **y**:   (n_y, m, T_y)
* **Wax**: (n_a, n_x)
* **Waa**: (n_a, n_a)
* **Wya**: (n_y, n_a)
* **ba**:  (n_a, 1)
* **by**:  (n_y, 1)

RecurrentNeuralNetwork attributes:
* **n_a** - dimension of hidden units

RecurrentNeuralNetwork.optimize method:
* **data** - list of batches e.g. [[x1, y1], [x2, y2], ...] where xi and yi are three dimensional numpy arrays (n_x/n_y, m, T_x)
* **epochs** - number of times to pass through data; default = 20
* **learn** - learning rate; default = 0.01
* **max_grad** - gradient clipping value; default = 10

### References 

Inspired by: [Deep Learning Specialisation](https://www.coursera.org/learn/nlp-sequence-models?specialization=deep-learning) on Coursera.
