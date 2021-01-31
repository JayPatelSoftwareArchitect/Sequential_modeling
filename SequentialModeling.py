import tensorflow as tf

# Download and import the MIT 6.S191 package
#!pip install mitdeeplearning
#!apt-get install abcmidi timidity > /dev/null 2>&1

import mitdeeplearning as mdl

import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
songs = mdl.lab1.load_training_data()

# Convert the ABC notation to audio file and listen to it
#mdl.lab1.play_song(example_song)

# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs) 
# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d", 
#   we can evaluate `char2idx["d"]`.  
char2idx = {u:i for i, u in enumerate(vocab)}
# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)
print('{')
for char,_ in zip(char2idx, range(len(vocab))):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

### Vectorize the songs string ###


def vectorize_string(string):
    arr = []
    for i in string:
        arr.append(char2idx[i])
    return np.array(arr,dtype=np.int32)

vectorized_songs = vectorize_string(songs_joined)


#create training examples

# Our next step is to actually divide the text into example sequences 
# that we'll use during training. Each input sequence that we 
# feed into our RNN will contain seq_length characters from the text.
#  We'll also need to define a target sequence for each input sequence, 
# which will be used in training the RNN to predict the next character.
#  For each input, the corresponding target will contain the same length 
# of text, except shifted one character to the right.

# To do this, we'll break the text into chunks of seq_length+1. 
# Suppose seq_length is 4 and our text is "Hello". Then, our
#  input sequence is "Hell" and the target sequence is "ello".

# The batch method will then let us convert this stream of character 
# indices to sequences of the desired size.

### Batch definition to create training examples ###

def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  idx = np.random.choice(n)
  #construct a list of input sequences for the training batch
  input_batch = vectorized_songs[idx+seq_length*batch_size:idx+seq_length*batch_size*2]
  #construct a list of output sequences for the training batch
  output_batch = vectorized_songs[idx+seq_length*batch_size+1:idx+seq_length*batch_size*2+1]
  
  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch =  np.reshape(input_batch, (batch_size,seq_length ))
  y_batch = np.reshape(output_batch, (batch_size,seq_length))
  return x_batch, y_batch

# Perform some simple tests to make sure your batch function is working properly! 
test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
   print("======\n[FAIL] could not pass tests")
else: 
   print("======\n[PASS] passed all tests!")


x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)


# Now we're ready to define and train a RNN model on our
#  ABC music dataset, and then use that trained model to 
# generate a new song. We'll train our RNN using batches
#  of song snippets from our dataset, which we generated in the 
# previous section.

# The model is based off the LSTM architecture,
# where we use a state vector to maintain information about 
# the temporal relationships between consecutive characters. 
# The final output of the LSTM is then fed into a fully connected
# Dense layer where we'll output a softmax over each character 
# in the vocabulary, and then sample from this distribution to 
# predict the next character.

# As we introduced in the first portion of this lab,
# we'll be using the Keras API, specifically, tf.keras.Sequential, 
# to define the model. Three layers are used to define the model:

#     tf.keras.layers.Embedding: This is the input layer,
#      consisting of a trainable lookup table that maps the numbers 
#      of each character to a vector with embedding_dim dimensions.
#     tf.keras.layers.LSTM: Our LSTM network, with size units=rnn_units.
#     tf.keras.layers.Dense: The output layer, with vocab_size outputs.


def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

### Defining the RNN Model ###
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #   of a fixed embedding size
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    # Call the LSTM function defined above to add this layer.
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    # into the vocabulary size. 
    #Add the Dense layer.
    tf.keras.layers.Dense(units=vocab_size, activation='relu')
  ])

  return model


# Build a simple model with default hyperparameters. You will get the 
#   chance to change these later.
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)

# It's always a good idea to run a few simple checks on our model to see that it behaves as expected.

# First, we can use the Model.summary function to print out a summary of our model's internal workings. Here we can check the layers in the model, the shape of the output of each of the layers, the batch size, etc.

model.summary()
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")


sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

# Now it's time to train the model!

# At this point, we can think of our next character prediction problem as a 
# standard classification problem. Given the previous state of the RNN, 
# as well as the input at a given time step, we want to predict the class of the
#  next character -- that is, to actually predict the next character.

# To train our model on this classification task, we can use a form of the 
# crossentropy loss (negative log likelihood loss). Specifically, we will use the 
# sparse_categorical_crossentropy loss, as it utilizes integer targets for categorical
#  classification tasks. We will want to compute the loss using the true targets -- the labels
#  -- and the predicted targets -- the logits.

### Defining the loss function ###
    #define the loss function to compute and return the loss between
    #the true labels and predictions (logits). Set the argument from_logits=True.
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) # TODO
  return loss

#compute the loss using the true next characters from the example batch 
#and the predictions from the untrained model several cells above
example_batch_loss = compute_loss(y, pred) 


print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())
# Let's start by defining some hyperparameters for training the model. To start, we have provided some reasonable values for some of the parameters. It is up to you to use what we've learned in class to help optimize the parameter selection here!


### Hyperparameter setting and optimization ###

# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


# Now, we are ready to define our training operation -- the optimizer and duration of training -- and use this function to train the model. You will experiment with the choice of optimizer and the duration for which you train your models, and see how these changes affect the network's output. Some optimizers you may like to try are Adam and Adagrad.

# First, we will instantiate a new model and an optimizer. Then, we will use the tf.GradientTape method to perform the backpropagation operations.

# We will also generate a print-out of the model's progress through training, which will help us easily visualize whether or not we are minimizing the loss.
### Define optimizer and training operation ###

#instantiate a new model for training using the `build_model`
#function and the hyperparameters created above.
model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)

#instantiate an optimizer with its learning rate.
#Checkout the tensorflow website for a list of supported optimizers.
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/
#Try using the Adam optimizer to start.
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function
def train_step(x, y): 
  # Use tf.GradientTape()
  with tf.GradientTape() as tape:
  
    #feed the current input into the model and generate predictions
    y_hat = model(x)
  
    #compute the loss!
    loss = compute_loss(y, y_hat)

  # Now, compute the gradients 
    #       complete the function call for gradient computation. 
    #       Remember that we want the gradient of the loss with respect all 
    #       of the model parameters. 
    #       HINT: use `model.trainable_variables` to get a list of all model
    #       parameters.
    
    grads = tape.gradient(loss, model.trainable_variables)
  
  # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    return loss


##################
# Begin training!#
##################

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for iter in tqdm(range(num_training_iterations)):

  # Grab a batch and propagate it through the network
  x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
  loss = train_step(x_batch, y_batch)

  # Update the progress bar
  history.append(loss.numpy().mean())
  print("scalar_loss:      ", loss.numpy().mean())

  plt.plot(history)
  plt.legend(('Predicted', 'True'))
  plt.xlabel('Iteration')
  plt.ylabel('x value')
  plt.show()


# Update the model with the changed weights!
  if iter % 100 == 0:     
    model.save_weights(checkpoint_prefix)
 
# Save the trained model and the weights
model.save_weights(checkpoint_prefix)
