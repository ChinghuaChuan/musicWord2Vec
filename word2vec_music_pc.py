# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code based on TensorFlow word2vec example, modified by Dorien Herremans and Ching-Hua Chuan as described in
# Chuan C.H., Agres, K., and Herremans D., "From Context to Concept: Exploring Semantic Relationships in Music with Word2Vec,"
# Neural Computing and Applications, special issue on Deep Learning for Music and Audio, Springer, 32(4), 2020, pp. 1023-1036,
# DOI: https://doi.org/10.1007/s0052. 
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import csv
from scipy.spatial import distance
#import matplotlib.pylab as plt


import itertools;
from collections import Counter

# Read the data into a list of strings.
def read_data():

  file_name = 'music_slice.txt'; #1800s_pitch_list.txt'; #'all_encoding.txt'

  with open(file_name, 'r') as f:
      raw_data = f.read()
      print("Data length:", len(raw_data))

  all_words = raw_data.split(' ')
  pieces = raw_data.split('\n')

  data = []
  for piece in pieces:
    slices = piece.split(' ')
    for slice in slices:
      if slice!='':
        data.append(slice)
  print("number of unique words: ", len(set(all_words)))

  return data

  
def read_pitch_dictionary():
  file_name='dictionary_pc_set.csv'
  pitch_dictionary = {}
  with open(file_name, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
      pitch_dictionary[row[0]]=row[1]
  return pitch_dictionary


words = read_data()
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 2000; 


def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.

print('Vocabulary has been built with size', vocabulary_size)

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  span = 2 * skip_window + 1  # [ skip_window target skip_window ]  Context that you predict

  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


# Step 4: Build and train a skip-gram model.

batch_size = 128 
embedding_size = 256   # Dimension of the embedding vector.
skip_window = 4       # How many words to consider left and right.
num_skips = skip_window*2        # How many times to reuse an input to generate a label.
alpha = 0.1   # learning rate
num_steps = 1000001   # how long to train for



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 5     # Random set of words to evaluate similarity on.
valid_window = 20  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

# Hand-picked validation set for music
valid_centers = ['145', '545', '2180'] # 145: C-E-G, 545: F-A-C, 2180: G-B-D 
valid_neighbors = [['2180', '545', '529', '1160', '290', '1156'], ['145', '1060', '548', '265', '1090', '137'], ['580', '145', '2192', '1060', '265', '548']] # V, IV, vi, IIIb, IIb, v



graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  #with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
  embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='embeddings')
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  # Construct the variables for the NCE loss
  nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True),name='norm')
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')

  # Add variable initializer.
  init = tf.global_variables_initializer()




# Step 5: Begin training.
with tf.Session(graph=graph) as session:
  saver = tf.train.Saver()
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  pitch_dictionary = read_pitch_dictionary()
  average_loss = 0
  results_C = []
  results_F = []
  results_G = []
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      # print("Average loss at step ", step, ": ", average_loss)
      print(average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    # get the 8  most closest to the validation set
    if step % 10000 == 0:
      sim = similarity.eval()
      norm_embeddings = normalized_embeddings.eval()
      
      # hand-picked validation examples
      for i, center in enumerate(valid_centers):
        center_i = dictionary[center]
        sim_values = []
        log_str = "Similarity to %s:" % pitch_dictionary[center]
        for neighbor in valid_neighbors[i]:
          neighbor_i = dictionary[neighbor]
          center_embedding = norm_embeddings[center_i]
          neighbor_embedding = norm_embeddings[neighbor_i]
          cos_dist = distance.cosine(center_embedding, neighbor_embedding)
          sim_values.append(cos_dist)
          log_str = "%s %s:%s," % (log_str, pitch_dictionary[neighbor], cos_dist)
        if i == 0:
          results_C.append(sim_values)
        elif i == 1:
          results_F.append(sim_values)
        else:
          results_G.append(sim_values)  
        print(log_str)
      
  final_embeddings = normalized_embeddings.eval()
  saver.save(session, "saves/word2vec_music_pc_train")
  np.savetxt('results_C.txt', results_C, delimiter=',', newline='\n')
  np.savetxt('results_F.txt', results_F, delimiter=',', newline='\n')
  np.savetxt('results_G.txt', results_G, delimiter=',', newline='\n')
  



# Step 6: Visualize the embeddings.
#
#
# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#   assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#   plt.figure(figsize=(18, 18))  # in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')
#
#   plt.savefig(filename)

