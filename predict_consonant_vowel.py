# modules

from recurrent_net import RecurrentNeuralNetwork
import numpy as np
from english_words import english_words_lower_alpha_set
import matplotlib.pyplot as plt

# characters

characters = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",\
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
    
# vowels

vowels = ("a", "e", "i", "o", "u")

# dimensions

n_x = len(characters)   # input features
n_y = 2                 # softmax output - vowel or consonant
T_x = T_y = 5           # number of timesteps
m = 32                  # batch size

# dictionaries

charToIdx = {}
idxToChar = {}
for i, char in enumerate(characters):
    charToIdx[char] = i
    idxToChar[i] = char

# extract words of length 5 from words corpus; ensure only valid characters

five_letters = []
for word in english_words_lower_alpha_set:
    if len(word) == T_x:
        valid = True
        for char in word:
            if char not in characters:
                valid = False
        if valid:
            five_letters.append(word)

# number of examples

len(five_letters)
five_letters[0:10]

# data in correct format

data = []

start_idx = 0
end_idx = m
while end_idx < len(five_letters):
    X = np.zeros((n_x, m, T_x))
    Y = np.zeros((n_y, m, T_y))
    for i in range(m):
        current_word = five_letters[start_idx + i]
        for j, char in enumerate(current_word):
            idx = charToIdx[char]
            X[idx,i,j] = 1
            if char in vowels:
                Y[0,i,j] = 1
            else:
                Y[1,i,j] = 1
    data.append([X, Y])
    
    start_idx += m
    end_idx += m
        
# train model to recognize vowels/consonants

model = RecurrentNeuralNetwork(n_a = 32)
loss_per_epoch, opt_params = model.optimize(data = data, epochs = 10, learn = 0.01, max_grad = 10)

# plot loss per epoch

plt.figure()
plt.plot(loss_per_epoch, 'r-o')
plt.xlabel("Epoch")
plt.ylabel("Loss")

# inference

def inference(word, params):
    # dimensions
    T_x = T_y = len(word)
    M = 1
    # populate x and y arrays
    x = np.zeros((n_x, M, T_x))
    y = np.zeros((n_y, M, T_y))
    for i in range(T_x):
        val = charToIdx[word[i]]
        x[val, 0, i] = 1
        if word[i] in ("a", "e", "i", "o", "u"):
            y[0, 0, i] = 1
        else:
            y[1, 0, i] = 1
    # forward propagate through network
    Y_hat, cache, error = model.rnn_forward(x, y, params)
    # print out results
    for i in range(T_x):
        if Y_hat[0,0,i] > 0.5:
            pred = "vowel"
        else:
            pred = "consonant"
        print(f"{word[i]} ---> {pred}")
    print("\n")

inference("asymptotic", opt_params)
inference("indignation", opt_params)
inference("hippopotamus", opt_params)
