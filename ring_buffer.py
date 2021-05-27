import numpy as np

class RingBuffer:

    def __init__(self, size, elem_dimension=None):
        
        self.size = size

        if(elem_dimension is not None):
            self.buffer = np.zeros(shape=(size,elem_dimension))
            self.TwoD = True
        else:
            self.buffer = np.zeros(shape=(size))
            self.TwoD = False

        self.idx = 0
        self.num_filled = 0

    def push(self, elem):
        if(self.TwoD):
            self.buffer[self.idx,:] = elem
        else:
            self.buffer[self.idx] = elem
        self.idx = (self.idx + 1)%self.size
        self.num_filled = min(self.num_filled+1, self.size)

    def get(self):

        if(self.num_filled == 0): return []

        if( self.TwoD ):
            return np.roll(self.buffer, (self.size-self.idx), axis=0)[-self.num_filled:,:]
        else:
            return np.roll(self.buffer, (self.size-self.idx), axis=0)[-self.num_filled:]

    def is_full(self):
        return (self.num_filled == self.size)

    def reset(self):
        self.idx = 0
        self.num_filled = 0
