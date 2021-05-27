import numpy as np 
import tensorflow as tf 
from global_vars import*

def get_histogram_cost(x,y):
    x = x + eps
    y = y + eps
    x = tf.math.l2_normalize(x) #Normalize
    y = tf.math.l2_normalize(y)
    xHist = tf.histogram_fixed_width(tf.math.log(tf.abs(x)+eps), [-20, 0]) #Compute histograms
    yHist = tf.histogram_fixed_width(tf.math.log(tf.abs(y)+eps), [-20, 0])
    cost = tf.reduce_sum(tf.math.square(xHist - yHist))
    return tf.cast(cost, tf.float32)

if __name__ == '__main__':
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    cost = get_histogram_cost(x,y)
    print(cost)