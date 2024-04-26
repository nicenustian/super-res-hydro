import tensorflow as tf
tfkl = tf.keras.layers
#import matplotlib.pyplot as plt

# class UpSample2D(tf.keras.Model):
#     def __init__(self, scale):
#         super(UpSample2D, self).__init__()
        
#         self.scale = scale

#     def call(self, inputs):
#         return tf.image.resize(inputs, size=(inputs.shape[1]*self.scale, 
#                                              inputs.shape[2]*self.scale),
#                                method='nearest', antialias=True)
    

class UpSample2D(tf.keras.Model):
    def __init__(self, scale):
        super(UpSample2D, self).__init__()
        self.scale = scale

    def call(self, inputs):
        
        # Repeat each row 'scale' times vertically
        repeated_rows = tf.repeat(inputs, repeats=self.scale, axis=1)
        
        # Repeat each column 'scale' times horizontally
        repeated_cols = tf.repeat(repeated_rows, repeats=self.scale, axis=2)
        
        return repeated_cols    


# inputs = tf.random.normal((2, 32, 32, 1))

# print('inputs = ', inputs.shape)
# upsample = UpSample2D(scale=2)
# outputs = upsample(inputs)
# print('outputs = ', outputs.shape)

# plt.subplot(2, 1, 1)
# plt.imshow(inputs[0,:,:,0].numpy())
# plt.subplot(2, 1, 2)
# plt.imshow(outputs[0,:,:,0].numpy())