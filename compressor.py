import numpy as np
import librosa as lib
from scipy.io.wavfile import write
from ring_buffer import*
import matplotlib.pyplot as plt
from global_vars import*
import tensorflow as tf
from cost_function import*

class Compressor:

    def __init__(self, thr, ratio, Fs = 44100):
        
        self.threshold = thr
        self.ratio = ratio
        self.last_gain = 0 #Maintains frame continuity
        self.tau_a = 1
        self.tau_r = 1
        self.Fs = Fs
        self.gain_history = RingBuffer(Fs*30)
        self.time = 0
        self.cost_frames = []
        self.generate_cost = False
        self.cost_frequency = tracking_freq
        self.set_times(10,100) #Initial attack/release setting

    def get_costs_and_reset(self):
        costs = self.cost_frames.copy()
        self.cost_frames.clear()
        return np.concatenate(costs, axis=0)

    def get_costs_and_reset_tf(self):
        return tf.numpy_function(self.get_costs_and_reset, (), [tf.float32])

    def set_times(self, attack, release):

        if attack < 1: attack = 1
        if release < 1: release = 1

        if attack > 1000: attack = 1000
        if release > 10000: release = 10000

        self.tau_a = np.exp( -np.log(9)/(self.Fs * attack/1000.0) )
        self.tau_r = np.exp( -np.log(9)/(self.Fs * release/1000.0) )

    def get_times(self):
        attack = (-np.log(9)/np.log( self.tau_a ))/self.Fs * 1000
        release = (-np.log(9)/np.log( self.tau_r ))/self.Fs * 1000
        return attack, release


    def set_times_tf(self, attack, release):
        return tf.numpy_function(self.set_times, (attack, release), []) 

    def process_frame(self, x, generate_cost):
        
        if(generate_cost): #Generate cost audio
            N = len(x)
            t = np.arange(self.time, self.time+N)
            x_cost = .25*np.sin(2*np.pi*self.cost_frequency*t/self.Fs)
            self.time += N

        x_raw = np.copy(x)
        x_dB = np.array([20 * np.log10( abs(n) + .000000001 ) for n in list(x) ]) #Convert frame to dB
        x_sc = np.array([ n if n < self.threshold else self.threshold + (n - self.threshold)/float(self.ratio) for n in x_dB ]) #Gain target
        gc = x_sc - x_dB #Gain

        if sum(abs(gc)) > 0: #Check if compressor is active
            active = 1
        else:
            active = 0
    
        #Gain smoothing:
        gs = np.zeros(len(gc))

        if gc[0] <= self.last_gain: #First sample
            gs[0] = self.tau_a * self.last_gain + (1-self.tau_a)*gc[0]

        else:
            gs[0] = self.tau_r * self.last_gain + (1-self.tau_r)*gc[0]

        self.gain_history.push(gs[0]) #Update gain history

        for n in range(1,len(gs)):
            if gc[n] <= gs[n-1]: 
                gs[n] = self.tau_a * gs[n-1] + (1-self.tau_a)*gc[n]

            else:
                gs[n] = self.tau_r * gs[n-1] + (1-self.tau_r)*gc[n]
            
            self.gain_history.push(gs[n]) #Update gain history

        self.last_gain = gs[-1] #Save last gain value

        #Linearize gain, and apply to input:
        g_lin = np.array( [10.0**(float(x)/20) for x in gs] )
        y = np.multiply(x_raw, g_lin)

        accuracy_cost = np.sum( np.abs(gc - gs) ) / len(x) #Penalizes slow convergence
        
        if(generate_cost):
            y_cost = np.multiply(x_cost, g_lin) #Apply gain curve to cost audio
            self.cost_frames.append(y_cost)
            
        return np.array(y, dtype='float32'), np.array(accuracy_cost, dtype='float32'), np.array(active, dtype='int16')

    def process_frame_tf(self, x, generate_cost):
        return tf.numpy_function(self.process_frame, [x, generate_cost], [tf.float32, tf.float32, tf.int16])

    def get_gain_history(self):
        return self.gain_history.get()

    def plot_gain_history(self, save_path=None):
        gain = self.gain_history.get()
        t = np.arange(0,len(gain))/self.Fs
        plt.plot(t,gain)
        plt.xlabel("Time (s)")
        plt.ylabel("Gain Reduction (dB)")
        plt.title("Gain Reduction History")
        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def get_state_tf(self):
        return tf.numpy_function(self.get_state, (), [tf.float32])
    
    def get_state(self):
        return np.array([self.tau_a, self.tau_r, self.last_gain], dtype='float32')