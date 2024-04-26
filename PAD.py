import tensorflow as tf
tfkl = tf.keras.layers

class PAD(tf.keras.Model):
    def __init__(self,  image=False, name='pad'):
        super(PAD, self).__init__(name=name)
        self.image = image
    
    def call(self, x):
        if not self.image:  # 1D input
            nearest_start = tf.expand_dims(x[:, 0, :], axis=1)
            nearest_end = tf.expand_dims(x[:, -1, :], axis=1)
            padded_x = tf.concat([nearest_start, x, nearest_end], axis=1)
        elif self.image:  # 2D input
            nearest_top_row = tf.expand_dims(x[:, 0, :, :], axis=1)
            nearest_bottom_row = tf.expand_dims(x[:, -1, :, :], axis=1)
            padded_x = tf.concat([nearest_top_row, x, nearest_bottom_row], axis=1)
            
            nearest_left_column = tf.expand_dims(padded_x[:, :, 0, :], axis=2)
            nearest_right_column = tf.expand_dims(padded_x[:, :, -1, :], axis=2)
            padded_x = tf.concat([nearest_left_column, padded_x, nearest_right_column], axis=2)

        return padded_x
    
# inputs = tf.random.normal((2, 32, 4))


# print('inputs = ',inputs.shape)
# pad = PAD()
# outputs = pad(inputs)
# print('outputs = ', outputs.shape)