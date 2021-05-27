import librosa as lib
import numpy as np
from scipy.io.wavfile import write
import tensorflow as tf

class AudioBuffer:

    def __init__(self, file_path, frame_len, lookahead_frames, Fs=44100):
        
        self.aud_in = lib.load(file_path, sr=Fs )[0] #Audio file (mono)
        self.aud_in = self.aud_in.astype('float32')
        self.N = int(np.ceil(len(self.aud_in)/frame_len)) * frame_len 
        self.aud_out = np.zeros(self.N, dtype='float32')

        self.Fs = Fs
        self.frame_len = frame_len 
        self.lookahead_len = frame_len * (lookahead_frames-1)
        
        self.aud_in = np.concatenate( (self.aud_in, np.zeros(self.N - len(self.aud_in)))  ) #Round up to nearest frame
        self.aud_in = np.concatenate( (np.zeros(self.lookahead_len, dtype='float32'), self.aud_in, np.zeros(self.lookahead_len, dtype='float32')) ) #Add zeros to allow lookahead
        
        self.idx_in = 0
        self.idx_out = 0

    def reset(self):
        self.idx_in = 0
        self.idx_out = 0
        self.aud_out = np.zeros(self.N)

    def get_raw_input(self):

        return self.aud_in[self.lookahead_len:self.lookahead_len+self.N] #Remove leading and trailing zeros
    
    def next_frame(self):
        
        input_frame = self.aud_in[ self.idx_in : self.idx_in+self.frame_len ]
        lookahead_frame = self.aud_in[ self.idx_in+self.lookahead_len : self.idx_in+self.lookahead_len+self.frame_len ]

        self.idx_in += self.frame_len
        done = int(self.idx_in >= self.N)

        return input_frame.astype('float32'), lookahead_frame.astype('float32'), np.array(done, dtype='int16')

    def next_frame_tf(self):
        return tf.numpy_function(self.next_frame, (), [tf.float32, tf.float32, tf.int16])

    def write_frame(self, frame):
        assert(len(frame) == self.frame_len)
        assert( (self.N - self.idx_out) >= self.frame_len)
        self.aud_out[self.idx_out:self.idx_out+self.frame_len] = frame
        self.idx_out += self.frame_len

    def write_output_to_file(self, filename):
        write(filename, self.Fs, self.aud_out)
        return (np.max(self.aud_out) >= 1) #Check clipping

    def get_file_len(self):
        return self.N
