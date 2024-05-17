import tensorflow as tf
tfkl = tf.keras.layers

class PAD(tf.keras.Model):
    def __init__(self,  image=False, name='pad'):
        super(PAD, self).__init__(name=name)
    
    def call(self, x):
        
        nearest_start = tf.expand_dims(x[:, 0, :], axis=1)
        nearest_end = tf.expand_dims(x[:, -1, :], axis=1)
        padded_x = tf.concat([nearest_start, x, nearest_end], axis=1)
        
        return padded_x
