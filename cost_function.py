from scipy.fft import fft
from scipy.signal import iirnotch, lfilter, square, stft, resample
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
from utils import*
from global_vars import*

cost_scaling = 1000

def get_rate_cost(cost_audio, weight_audio, frame_len, Fs=44100):

    '''Computes rate cost from compressor's cost audio signal
    \ncost_audio -> cost audio array from compressor
    \nweight_audio -> original audio signal used to weight costs
    \nframe_len -> number of samples per frame
    \nFs -> sampling rate'''

    #Filter out fundamental
    [b,a] = iirnotch(tracking_freq,1.5,Fs) 
    filtered_audio = lfilter(b,a,cost_audio)

    _, _, cost_spectrum = stft(filtered_audio, fs=Fs, nperseg=frame_len, noverlap=0, nfft=max(1024, frame_len)) #STFT
    
    cost_spectrum = cost_spectrum[:,:-1] #Truncate last frame to match length
    cost_spectrum = np.abs(cost_spectrum) #Make real
    cost = np.sum( np.abs(cost_spectrum), axis=0 ) #Sum harmonic energy
    cost[0:2] = np.reshape([0,0], (-1,)) #Removes anomalies at beginning to audio file
    
    weight = weight_audio ** 2 #Energy
    weight = weight/np.max(weight) #Normalize
    
    #Compute average weight for each frame
    averaged_weight = []
    for i in range(0, len(weight), frame_len):
        averaged_weight.append( np.log( (1/frame_len) * np.sum( weight[i : i+frame_len] ) + 1) )  

    return cost_scaling * np.array(cost, dtype='float32') * np.array(averaged_weight, dtype='float32')

def get_rate_cost_tf( cost_audio, weight_audio, frame_len ):
    return tf.numpy_function(get_rate_cost, (cost_audio, weight_audio, frame_len), [tf.float32])

def get_total_reward( rate_costs, accuracy_costs, weights ):
    cost = tf.math.add(rate_costs, weights[1]*accuracy_costs)
    reward = (1/(cost+.01))
    reward = tf.cast(reward, 'float32')
    return reward

def get_total_cost( rate_costs, accuracy_costs, weights ):
    costs = tf.math.add(rate_costs, weights[1]*accuracy_costs)
    costs = tf.cast(costs, 'float32')
    return costs

def get_total_reward_tf( rate_costs, accuracy_costs, weights ):
    return tf.numpy_function(get_total_reward, (rate_costs, accuracy_costs, weights), [tf.float32])