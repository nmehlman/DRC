from tensorflow.keras.losses import MSE
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv1D, MaxPool1D, Flatten, Reshape, BatchNormalization
from numpy import argmax, array, concatenate
from tensorflow.keras.models import load_model
from numpy.random import randn
from tensorflow import random
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
from numpy import max, abs, concatenate, sum, argmax
from features import EnvelopeFinder
import os
from global_vars import*
from audio_buffer import*
from compressor import Compressor
from features import*
from cost_function import*
from utility import time_str
import matplotlib.pyplot as plt
from timer import runTimer
from utils import directory_init, idx_to_time
from sklearn.preprocessing import StandardScaler
import pickle

'''Defines model class for Policy Gradient implementation'''

class PGModel:

    '''
    Model for implementing Policy Gradient algorithm.
    \nhistory_neurons -> number of neurons for history section
    \nlookahead_neurons -> number of neurons for lookahead section
    '''

    def __init__(self, history_neurons, lookahead_neurons):

        if compState: #Use compressor state in place of 'history' audio frames
            self.history_len = 3
            self.lookahead_len = lookahead_neurons
            history_neurons = 3

        else: #Use output audio frames 
            self.history_len = history_neurons
            self.lookahead_len = lookahead_neurons

        i1 = Input(shape=(self.history_len,))
        i2 = Input(shape=(self.lookahead_len,))

        i = Concatenate(axis=1)([i1,i2])
        x = Reshape(( history_neurons+lookahead_neurons, 1))(i) #Needed to make shape comform to convolution layer

        ## ATTACK NETWORK ##
        x1 = Conv1D(128, 4, activation='relu')(x)
        x1 = MaxPool1D()(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(256, 4, activation='relu')(x1)
        x1 = MaxPool1D()(x1)
        x1 = BatchNormalization()(x1)
        x1 = Flatten()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(num_actions)(x1)
        
        ## RELEASE NETWORK ##
        x2 = Conv1D(128, 4, activation='relu')(x)
        x2 = MaxPool1D()(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(256, 4, activation='relu')(x2)
        x2 = MaxPool1D()(x2)
        x2 = BatchNormalization()(x2)
        x2 = Flatten()(x2)
        x2 = Dense(256,activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dense(num_actions)(x2)
        
        self.actor = Model(inputs=[i1,i2], outputs=[x1,x2])

        ## CRITIC (VALUE) NETWORK ##
        x = Conv1D(64, 4, activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(16, 4, activation='relu')(x)
        x = MaxPool1D()(x)
        x = Flatten()(x)
        x = Dense(64)(x)
        x = Dense(64)(x)
        x = Dense(1, activation='relu')(x)

        self.critic = Model(inputs=[i1,i2], outputs=x)

    def __call__(self, state):

        '''Merges actor and critic predictions
        \nstate -> state from envelope finder = [...history ... lookahead...]
        \ngreedy -> predicts highest-valued a action if true, else samples from distribution'''

        attack_idx, release_idx, attack_prob, release_prob = self.predict_actor(state)
        value = self.predict_critic(state) #Get value
        attack = idx_to_time(attack_idx, attack_max) #Convert to ms
        release = idx_to_time(release_idx, release_max) #Convert to ms
        return attack, release, attack_prob, release_prob, value

    def predict_tf(self, state):
        
        history, lookahead = self.split_state_comp(state)
        
        #Actor predictions
        predictions = self.actor([history, lookahead])
        attack_idx = random.categorical(predictions[0], 1)[0,0]
        release_idx = random.categorical(predictions[1], 1)[0,0]
        attack_prob = tf.nn.softmax(predictions[0])[0][attack_idx]
        release_prob = tf.nn.softmax(predictions[1])[0][release_idx]

        #Critic predictions
        value = self.critic([history, lookahead])[0]

        attack = idx_to_time_tf(attack_idx, attack_max) #Convert to ms
        release = idx_to_time_tf(release_idx, release_max) #Convert to ms

        return tf.cast(attack, 'float32'), tf.cast(release, 'float32'), tf.cast(attack_prob, 'float32'), tf.cast(release_prob, 'float32'), tf.cast(value, 'float32')

    def predict_actor(self, state):

        '''Predict attack and release value from actor network
        \nstate -> state from envelope finder = [...history ... lookahead...]
        \ngreedy -> predicts highest-valued a action if true, else samples from distribution'''

        history, lookahead = self.split_state(state)
        predictions = self.actor([history, lookahead])
        attack_idx = random.categorical(predictions[0],1)[0,0]
        release_idx = random.categorical(predictions[1],1)[0,0]
        attack_prob = tf.nn.softmax(predictions)[0][0][attack_idx]
        release_prob = tf.nn.softmax(predictions)[1][0][release_idx]

        return attack_idx, release_idx, attack_prob, release_prob  
        
    def predict_critic(self, state):

        '''Predict state value using critic network
        \nstate -> state from envelope finder = [...history ... lookahead...]'''

        history, lookahead = self.split_state(state)
        value = self.critic([history, lookahead])[0]
        return value

    def split_state(self, state):

        '''Separates state into history and lookahead components
        \nstate -> state from envelope finder'''
        if compState:
            state = tf.expand_dims(state, 0)
            history = state[:,0:3]
            lookahead = state[:, -self.lookahead_len:]
        else:
            state = tf.expand_dims(state, 0)
            history = state[:, 0:self.history_len]
            lookahead = state[:, -self.lookahead_len:]

        return history, lookahead

    def split_state_comp(self, state):

        '''Separates state into history and lookahead components
        \nstate -> state from envelope finder'''
        state = tf.expand_dims(state, 0)
        history = state[:,0:3]
        lookahead = state[:, -self.lookahead_len:]

        return history, lookahead

    def save(self, save_path):
        self.actor.save(save_path + "/actor")
        self.critic.save(save_path + "/critic")

    def load(self, load_path):
        self.actor = tf.keras.models.load_model(load_path + '/actor')
        self.critic = tf.keras.models.load_model(load_path + '/critic')

def build_actor_critic(history_neurons, lookahead_neurons):

    if compState: #Use compressor state in place of 'history' audio frames
        history_len = 3
        lookahead_len = lookahead_neurons
        history_neurons = 3

    else: #Use output audio frames 
        history_len = history_neurons
        lookahead_len = lookahead_neurons

    i1 = Input(shape=(history_len,))
    i2 = Input(shape=(lookahead_len,))

    i = Concatenate(axis=1)([i1,i2])
    x = Reshape(( history_neurons+lookahead_neurons, 1))(i) #Needed to make shape comform to convolution layer

    ## ATTACK NETWORK ##
    x1 = Conv1D(128, 4, activation='relu')(x)
    x1 = MaxPool1D()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(256, 4, activation='relu')(x1)
    x1 = MaxPool1D()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(num_actions)(x1)
        
    ## RELEASE NETWORK ##
    x2 = Conv1D(128, 4, activation='relu')(x)
    x2 = MaxPool1D()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(256, 4, activation='relu')(x2)
    x2 = MaxPool1D()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)
    x2 = Dense(256,activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(num_actions)(x2)
        
    actor = Model(inputs=[i1,i2], outputs=[x1,x2])

    ## CRITIC (VALUE) NETWORK ##
    x = Conv1D(64, 4, activation='relu')(x)
    x = MaxPool1D()(x)
    x = Conv1D(16, 4, activation='relu')(x)
    x = MaxPool1D()(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dense(64)(x)
    x = Dense(1, activation='relu')(x)

    critic = Model(inputs=[i1,i2], outputs=x)

    return actor, critic

def split_state(state, lookahead_len, history_len):

    if compState:
        state = tf.expand_dims(state, 0)
        history = state[:,0:3]
        lookahead = state[:, lookahead_len:]

    else:
        state = tf.expand_dims(state, 0)
        history = state[:, 0:history_len]
        lookahead = state[:, -lookahead_len:]

    return history, lookahead

def predict_times(state, actor, critic):
        
    history, lookahead = split_state(state)
        
    #Actor predictions
    predictions = actor([history, lookahead])
    attack_idx = random.categorical(predictions[0], 1)[0,0]
    release_idx = random.categorical(predictions[1], 1)[0,0]
    attack_prob = tf.nn.softmax(predictions[0])[0][attack_idx]
    release_prob = tf.nn.softmax(predictions[1])[0][release_idx]

    #Critic predictions
    value = critic([history, lookahead])[0]

    attack = idx_to_time_tf(attack_idx, attack_max) #Convert to ms
    release = idx_to_time_tf(release_idx, release_max) #Convert to ms

    return tf.cast(attack, 'float32'), tf.cast(release, 'float32'), tf.cast(attack_prob, 'float32'), tf.cast(release_prob, 'float32'), tf.cast(value, 'float32')