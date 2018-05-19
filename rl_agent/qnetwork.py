import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Qnetwork():
    def __init__(self,field_size):
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)