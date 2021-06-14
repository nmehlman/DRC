import numpy as np
from global_vars import*
from os import mkdir
from utility import time_str
from matplotlib import pyplot as plt
from os.path import join
import tensorflow as tf

def moving_average(x, w, mode='valid'):
    return np.convolve(x, np.ones(w), mode)

def directory_init(t_str=None):

    if not t_str:
        t_str = time_str(sec=True, year=False, isPath=True)

    log_dir = "Logs" + t_str 
    mkdir(log_dir)

    write_path = join(log_dir, write_dir)
    time_plot_path = join(log_dir, time_plot_dir)
    save_path = join(log_dir, save_dir)
    cost_path = join(log_dir, cost_dir)
    gr_path = join(log_dir, gr_dir)
    
    mkdir(write_path)
    mkdir(time_plot_path)
    mkdir(cost_path)
    mkdir(gr_path)

    return save_path, time_plot_path, write_path, cost_path, gr_path, log_dir 

def idx_to_time(idx, t_max):
    '''Converts index of output neuron to associated time value
    \nidx -> output neuron index
    \nt_max -> maximum time value'''
    return np.array((idx/num_actions) * t_max, dtype='float32')

def idx_to_time_tf(idx, t_max):
    return tf.numpy_function(idx_to_time, (idx, t_max), [tf.float32])

def cost_plot(rate_costs, accuracy_costs, path):
    plt.figure(figsize=(8,8))
    t = np.arange(tf.shape(rate_costs)[0]) * frame_len/Fs

    plt.subplot(2,1,1)
    plt.plot(t,rate_costs)
    plt.title('Rate Cost')
    plt.ylabel('Cost')

    plt.subplot(2,1,2)
    plt.plot(t,accuracy_costs)
    plt.title('Accuracy Cost')
    plt.savefig(path)
    plt.close()

def loss_reward_plot(loss, reward, t_str):
    
    plt.figure(figsize=(8,8))

    plt.subplot(2,1,1)
    plt.plot(reward)
    plt.title('Rewards and Losses')
    plt.ylabel('Reward')
    
    plt.subplot(2,1,2)
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig('Logs'+t_str+'/Loss Reward Plot')
    plt.close()

def attack_release_plot(attack_times, release_times, save_path=None):

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    t = np.arange(len(attack_times)) * frame_len/Fs
    plt.plot(t, attack_times)
    plt.title('Attack Times')

    plt.subplot(2,1,2)
    plt.plot(t, release_times)
    plt.title('Release Times')

    if(save_path is not None):
        plt.savefig(save_path)
        
    else:
        plt.show()

    plt.close()

def make_plots(time_plot_path, cost_path, gr_path, epoch_number, 
histogram_costs, accuracy_costs, attack_times, release_times):

    cost_plot(histogram_costs, accuracy_costs, #Costs
    cost_path + "/epoch_{}".format(epoch_number+1))
    attack_release_plot(attack_times, release_times, time_plot_path + "/epoch_{}".format(epoch_number+1)) #Attack and release times
