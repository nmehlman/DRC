import numpy as np
import tensorflow as tf

lr = .001 #Learning rates
epochs = 500 #Number of training epochs
compState = True #Use compressor for state info

num_actions = 20
attack_max = 100
release_max = 1000

epsilon = .1 #For Epsilon Greedy
cost_weights = tf.constant([1, 1], dtype='float32') #[rate, accuracy]
cost_weights = tf.math.l2_normalize(cost_weights) #Normalize weights
eps = np.finfo(np.float32).eps.item() #Small number to avoid division by zero
gamma = .5 #Cost discounting
Fs = 44100 #Sample rate (Hz)
frame_len = 11025 #Samples per frame

lookahead_frames = 6 #Number of future frames fed to network 
history_frames = 25 #Number past frames fed to network
history_neurons = 256 #Input neurons for history portion
lookahead_neurons = 64 #Input neurons for lookahead_frame portion

if compState:
    state_len = 3 + lookahead_neurons #Length of full state
else:
    state_len = history_neurons + lookahead_neurons #Length of full state
    
thr = -30 #Compressor threshold
ratio = 3 #Compressor ratio
tracking_freq = 40

smooth_len = 10 #Time smoothing

write_dir = "Output Files"
time_plot_dir = "Time Plots"
save_dir = "Checkpoints"
cost_dir = "Costs"
gr_dir = "Gain Reduction"


