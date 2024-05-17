import tensorflow as tf
from PAD import PAD
tfkl = tf.keras.layers

'''
Inputs 1D signal with N-channels/features
Output batch x 1  numbers to represent the score for each example, probability
if adversial is True
'''

class ConvLayerDiscriminator(tf.keras.Model):
    def __init__(self, original_dim, filters, scale, initializer, num_features):
        
        super(ConvLayerDiscriminator, self).__init__()

        self.initializer = initializer
        self.num_features = num_features
        self.original_dim = original_dim
        self.conv_layers = []
    
        
        # NOTE: use of groups in Convolutions don't give correlated fields
        self.conv_layers.append(PAD())
        self.conv_layers.append(tfkl.Conv1D(
                        filters, kernel_size=3, strides=scale,
                        padding="valid", kernel_initializer=self.initializer,
                        ))

        self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))
        self.conv_layers.append(tfkl.Dropout(0.2))


    def call(self, inputs):
        x = inputs

        for layer in self.conv_layers.layers:
            x = layer(x)
                  
        return x



class DiscriminatorScale(tf.keras.Model):
    def __init__(self, original_dim, filters, scales,
                 initializer, num_features):
        
        super(DiscriminatorScale, self).__init__()
                
        self.conv_layers = []
        
        for num_filter, scale in zip(filters, scales):
            
            self.conv_layers.append(ConvLayerDiscriminator(
                original_dim, num_filter, scale, initializer,
                num_features))


    def call(self, inputs): 
        x = inputs
            
        for layer in self.conv_layers.layers:
            x = layer(x)
            
        return x



class Discriminator(tf.keras.Model):
    def __init__(self, original_dim, filters_list, scales_list, initializer, 
                 num_features, adversial=False):
        
        super(Discriminator, self).__init__()
        
        self.conv_layers = []
        self.output_layers = []
        self.original_dim = original_dim
        
        for filters, scales in zip(filters_list, scales_list):
            self.conv_layers.append(DiscriminatorScale(
                original_dim, filters, scales, initializer, num_features))
                    
        if adversial:
            self.flat = tfkl.Flatten()
            self.drop = tfkl.Dropout(0.2)
            self.dense  = tfkl.Dense(1, activation='sigmoid', 
                                            kernel_initializer=initializer)
        else:
            self.flat =tfkl.Flatten()
            self.drop = tfkl.Dropout(0.2)
            self.dense  = tfkl.Dense(1, kernel_initializer=initializer)
                

    def call(self, inputs):
                
        x = inputs
        for index, layer in enumerate(self.conv_layers.layers):
            x = layer(x)
                
        return self.dense(self.drop(self.flat(x)))
