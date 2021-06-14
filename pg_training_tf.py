from pg import*
from global_vars import*
from pg_model import*
from os import mkdir
from beepy import beep
import matplotlib.pyplot as plt
from audio import loadAudio
import os
from utils import directory_init, loss_reward_plot, make_plots
from utility import time_str
from shutil import copyfile, rmtree
from sys import argv
import sys
from traceback import print_exc
from glob import glob
from random import choice
from time import perf_counter
import tensorboard

'''Main script for running Policy Gradient Training Algorithm in Tensorflow Graph.'''

scaling = False #Scale loss/reward between files

logdir = 'tb_logs'

t_str = time_str(sec=True, year=False, isPath=True)
save_path, time_plot_path, write_path, cost_path, gr_path, log_dir = directory_init(t_str) #Create directories
copyfile('global_vars.py', log_dir + '/Parameters_PG') #Log parameter settings
train_dir = '../Training Sets/RL Tracks/*.wav' #Directory of training files
train_files = glob(train_dir) #Collect all file names
read_path = '../Training Sets/RL Tracks/rl3.wav' #Use specific training file

if compState: #Select scalar
        scalar_path = 'scalar_pickle_comp'
else:
    scalar_path = 'scalar_pickle'

actor, critic = build_actor_critic(history_neurons, lookahead_neurons)
audio = tf.constant( loadAudio(read_path, makemono=True)[0], dtype='float32')

opt = tf.keras.optimizers.Adam(learning_rate=lr)

if scaling:
    reward_scaling = {}
    loss_scaling = {}
    for f in train_files: #Create per-file scaling
        reward, loss = train_step(actor, critic, f, write_path, time_plot_path, cost_path, gr_path, lr, 0) #Run one training step
        reward_scaling[f] = 1/reward
        loss_scaling[f] = 1/loss
else:
    reward_scaling = dict.fromkeys(train_files, tf.constant(1.0, dtype='float32'))
    loss_scaling = dict.fromkeys(train_files, tf.constant(1.0, dtype='float32'))

reward_history = []
loss_history = []

try:
    for epoch in range(epochs):
        
        start = perf_counter()
        if epoch==0:
            writer = tf.summary.create_file_writer(logdir)
            tf.summary.trace_on(graph=True, profiler=True)

        episode_reward, episode_loss, plot_data = train_step_tf(actor, critic, audio, thr, ratio, opt, gamma) #Run one training step
        
        if epoch==0:
            with writer.as_default():
                tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=logdir)

        make_plots(time_plot_path, cost_path, gr_path, epoch, *plot_data)
        episode_reward = episode_reward * reward_scaling[read_path] #Normalize reward
        episode_loss = episode_loss * loss_scaling[read_path] #Normalize loss
        reward_history.append(episode_reward)
        loss_history.append(episode_loss)
        #model.save(save_path)
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

