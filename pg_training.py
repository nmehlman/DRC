from pg import*
from global_vars import*
from pg_model import*
from os import mkdir
from beepy import beep
import matplotlib.pyplot as plt
import os
from utils import directory_init, loss_reward_plot
from utility import time_str
from shutil import copyfile, rmtree
from sys import argv
import sys
from traceback import print_exc
from glob import glob
from random import choice
from time import perf_counter

'''Main script for running Policy Gradient Training Algorithm.'''

scaling = False #Scale loss/reward between files

t_str = time_str(sec=True, year=False, isPath=True)
save_path, time_plot_path, write_path, cost_path, gr_path, log_dir = directory_init(t_str) #Create directories
copyfile('global_vars.py', log_dir + '/Parameters_PG') #Log parameter settings
train_dir = '../Training Sets/RL Tracks/*.wav' #Directory of training files
train_files = glob(train_dir) #Collect all file names

model = PGModel(history_neurons, lookahead_neurons) #Build new model
#model.load('Logs/Wed Mar 24 13_40_04/Checkpoints') #Load saved model

if scaling:
    reward_scaling = {}
    loss_scaling = {}
    for f in train_files: #Create per-file scaling
        reward, loss = train_step(model, f, write_path, time_plot_path, cost_path, gr_path, lr, 0) #Run one training step
        reward_scaling[f] = 1/reward
        loss_scaling[f] = 1/loss
else:
    reward_scaling = dict.fromkeys(train_files, 1)
    loss_scaling = dict.fromkeys(train_files, 1)

reward_history = []
loss_history = []
try:
    for epoch in range(epochs):
        start = perf_counter()
        #read_path = choice(train_files) #Pick random training file
        read_path = '../Training Sets/RL Tracks/rl2.wav' #Use specific training file
        episode_reward, episode_loss = train_step(model, read_path, write_path, time_plot_path, cost_path, gr_path, lr, epoch) #Run one training step
        episode_reward = episode_reward * reward_scaling[read_path] #Normalize reward
        episode_loss = episode_loss * loss_scaling[read_path] #Normalize loss
        reward_history.append(episode_reward)
        loss_history.append(episode_loss)
        model.save(save_path)
        loss_reward_plot(loss_history, reward_history, t_str)
        end = perf_counter()
        print("Epoch {} completed: Time {:.2f} s ........... Reward: {:.4f} ........... Loss: {:.2f}".format(epoch+1, end-start, episode_reward, episode_loss))

    beep(1)

except Exception as E:
    beep(1)
    print_exc(file=sys.stdout)
    sys.exit()

finally:
    print('Terminating')
    if len(argv)>1 and argv[1] == 'NOLOG':
        rmtree(log_dir)

