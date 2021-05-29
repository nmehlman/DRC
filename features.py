import librosa as lib
import numpy as np
from scipy.signal import hilbert, resample
from ring_buffer import*
import matplotlib.pyplot as plt
import pickle
from global_vars import compState
from audio_buffer import AudioBuffer
import tensorflow as tf

'''Code for feature extraction from audio signals.'''

class EnvelopeFinder:

    '''Uses Hilbert Transform to extract envelope. Frames a fed one-by-one, and ring-buffers are
    used to preserve a given number of previous frames. History and lookahead components are independent.'''

    def __init__(self, history_frames, lookahead_frames, frame_len, history_neurons, lookahead_neurons, Fs=44100, scalar_path=None):
        
        '''Initilizes EnvelopeFinder instance.
        \nhistory_frames -> number of frames in history component. Used to set size of ring buffer.
        \nlookahead_frames -> number of frames in lookahead component. Used to set size of ring buffer.
        \nframe_len -> number of samples ber frame.
        \nhistory_neurons -> number of neurons in history input to network. Envelope is resampled to match this shape.
        \nlookahead_neurons -> number of neurons in lookahead input to network. Envelope is resampled to match this shape.
        \nFs -> sample rate
        \nscalar_path -> path to pickled instance of StandardScalar used to scale transformed envelope.'''

        self.history_neurons = history_neurons
        self.lookahead_neurons = lookahead_neurons
        self.Fs = Fs

        self.history_buffer = RingBuffer( history_frames*frame_len ) #Buffer for history 
        self.lookahead_buffer = RingBuffer( lookahead_frames*frame_len ) #Buffer for lookahead
        
        self.full = False #Whether or not buffers can be read from

        if scalar_path: self.scalar = pickle.load(open(scalar_path,'rb')) #Load StandardScalar
        
        else: self.scalar = None

    def reset(self):

        '''Resets buffers.'''

        self.history_buffer.reset()
        self.lookahead_buffer.reset()
        self.full = False

    def update_comp(self, lookahead_frame):

        '''Returns envelope 'state', after adding newest frames.
        \nhistory_frame -> latest history frame
        \nlookahead_frame -> latest lookahead frame'''
            
        #Add frames
        for i in lookahead_frame:
            self.lookahead_buffer.push(i)
            
        self.full = self.history_buffer.is_full() #Check if buffers are full
        #Get envelope
        lookahead_env = abs(hilbert( self.lookahead_buffer.get() ))
            
        #Resample so length matches network input
        lookahead_env_smoothed = resample(lookahead_env, self.lookahead_neurons)
            
        state = np.array(lookahead_env_smoothed)

        if self.scalar: #Scale
            state = self.scalar.transform([state])

        return state.flatten().astype('float32'), np.array((self.full or compState), dtype='int16')

    def update(self, history_frame, lookahead_frame):

        '''Returns envelope 'state', after adding newest frames.
        \nhistory_frame -> latest history frame
        \nlookahead_frame -> latest lookahead frame'''
            
        #Add frames
        for i in history_frame: 
            self.history_buffer.push(i)
        for i in lookahead_frame:
            self.lookahead_buffer.push(i)
            
        self.full = (self.history_buffer.is_full() and self.lookahead_buffer.is_full()) #Check if buffers are full

        #Get envelope
        history_env = abs(hilbert( self.history_buffer.get() ))
        lookahead_env = abs(hilbert( self.lookahead_buffer.get() ))
            
        #Resample so length matches network input
        history_env_smoothed = resample(history_env, self.history_neurons)
        lookahead_env_smoothed = resample(lookahead_env, self.lookahead_neurons)
            
        if compState: #Only return lookahead portion
            state = np.array(lookahead_env_smoothed)
            
        else: #Return lookahead and history
            state = np.concatenate( (history_env_smoothed, lookahead_env_smoothed) )

        if self.scalar: #Scale
            state = self.scalar.transform([state])

        return state.flatten().astype('float32'), np.array((self.full or compState), dtype='int16')

    def update_tf(self, lookahead_frame):
        return tf.numpy_function(self.update_comp, [lookahead_frame], [tf.float32, tf.int16])

if __name__ == '__main__':

    compState = False
    frame_len = 1000
    history_frames = 10
    lookahead_frames = 10
    history_neurons = 250
    lookahead_neurons = 250
    audio = AudioBuffer('../Training Sets/Test Files/Drums 1.wav', frame_len, lookahead_frames)
    env = EnvelopeFinder(history_frames, lookahead_frames, frame_len, history_neurons, lookahead_neurons)

    done = False
    while not done:
        i_frame, l_frame, done = audio.next_frame()
        state, full = env.update(i_frame, l_frame)
        if full:
            hist = state[:history_neurons]
            look = state[-lookahead_neurons:]
            plt.figure(figsize=(8,8))
            a1 = plt.subplot(2,1,1)
            a1.title.set_text('History')
            plt.plot(hist)
            a1.set_ylim([0,1])
            
            a2 = plt.subplot(2,1,2)
            a2.title.set_text('Lookahead')
            plt.plot(look)
            a2.set_ylim([0,1])
            plt.show(block=False)
            if input('') == 'q':
                done =True
            plt.close()