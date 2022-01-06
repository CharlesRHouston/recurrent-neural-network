# modules

from recurrent_net import RecurrentNeuralNetwork
import numpy as np
from english_words import english_words_lower_alpha_set
import matplotlib.pyplot as plt

# characters

characters = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",\
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "<S>", "<E>")

# dimensions

n_x = n_y = len(characters)     # features
word_len = 5                    # length of words to train on
T_x = T_y = word_len + 1        # number of timesteps
m = 32                          # batch size

# dictionaries

charToIdx = {}
idxToChar = {}
for i, char in enumerate(characters):
    charToIdx[char] = i
    idxToChar[i] = char

# extract words of length 5 from words corpus; ensure only valid characters

five_letters = []
for word in english_words_lower_alpha_set:
    if len(word) == word_len:
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
        X[charToIdx["<S>"],i,0] = 1
        Y[charToIdx["<E>"],i,-1] = 1
        for j, char in enumerate(current_word):
            idx = charToIdx[char]
            X[idx,i,j+1] = 1
            Y[idx,i,j] = 1
    data.append([X, Y])
    
    start_idx += m
    end_idx += m
        
# train model to recognize vowels/consonants

model = RecurrentNeuralNetwork(n_a = 32)
loss_per_epoch, opt_params = model.optimize(data = data, epochs = 200, learn = 0.001, max_grad = 10)

# plot loss per epoch

plt.figure()
plt.plot(loss_per_epoch, 'r-')
plt.xlabel("Epoch")
plt.ylabel("Loss")

# obtain the probability of a specific word

def word_probability(word, params):
    total_prob = 1
    X = word_to_one_hot(word)
    Y = X
    Y_hat, _, _ = model.rnn_forward(X, Y, params)
    for i, char in enumerate(word):
        idx = charToIdx[char]
        total_prob *= Y_hat[idx,0,i]
    print(f"Probability of '{word}' = {'{:0.2e}'.format(total_prob)}")
    # return total_prob

# convert a word to one-hot form

def word_to_one_hot(word):
    one_hot = np.zeros((len(characters), 1, len(word) + 1))
    one_hot[charToIdx["<S>"], 0, 0] = 1
    for i, char in enumerate(word):
        idx = charToIdx[char]
        one_hot[idx,0,i+1]
    return one_hot

# view probabilities of different words

word_probability("vague", opt_params)
word_probability("vageu", opt_params)
word_probability("qqqqq", opt_params)
word_probability("crate", opt_params)
word_probability("steak", opt_params)
word_probability("sttkk", opt_params)
word_probability("chloe", opt_params)
word_probability("light", opt_params)

# randomly generate words with the language model

def sample_network(params):
    word = ""
    while len(word) < word_len:
        X = word_to_one_hot(word)
        Y = X
        Y_hat, _, _ = model.rnn_forward(X, Y, params)
        idx = np.random.choice(len(characters), p = Y_hat[:,0,-1])
        # idx = np.argmax(Y_hat[:,0,-1])
        word = word + characters[idx]
    return word

print("\nRandomly sampled words:")
for _ in range(10):
    print(sample_network(opt_params))
