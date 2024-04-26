import tensorflow as tf
from PAD import PAD
tfkl = tf.keras.layers

'''
Inputs 1D signal with N-channels /  2D images both with N-channels
Output batch x 1  numbers to represent the score for each example, probability
if adversial is True
'''

class ConvLayerDiscriminator(tf.keras.Model):
    def __init__(self, original_dim, filters, scale,
                 initializer, num_features,
                 image=False):
        
        super(ConvLayerDiscriminator, self).__init__()

        self.image = image
        self.initializer = initializer
        self.num_features = num_features
        self.image = image
        self.original_dim = original_dim
        self.conv_layers = []
    
        
        # NOTE: use of groups in Convolutions don't give correlated fields
        if self.image:
            self.conv_layers.append(PAD(self.image))
            self.conv_layers.append(tfkl.Conv2D(
                        filters,# input_shape=(self.original_dim, num_features), 
                        kernel_size=3, strides=scale, padding="valid", 
                        kernel_initializer=self.initializer))
            
            self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))
            self.conv_layers.append(tfkl.Dropout(0.2))

           
        else:
            self.conv_layers.append(PAD(self.image))
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
                 initializer, num_features, image=False):
        
        super(DiscriminatorScale, self).__init__()
                
        self.conv_layers = []
        self.conv1x1_layers = []
        
        for num_filter, scale in zip(filters, scales):
            
            self.conv_layers.append(ConvLayerDiscriminator(
                original_dim, num_filter, scale, initializer,
                num_features, image=image))
            
            
        # # The 1 x 1 convolution
        # self.conv1x1_layers.append(PAD(image))
        # self.conv1x1_layers.append(tfkl.Conv1D(num_features, kernel_size=3, 
        #                                           strides=1, padding="valid",
        #                                           kernel_initializer=initializer,
        #                                           ))
        
        # self.conv1x1_layers.append(tfkl.LeakyReLU(alpha=0.2))
        # self.conv1x1_layers.append(tfkl.Dropout(0.2))


    def call(self, inputs): 
        x = inputs
            
        for layer in self.conv_layers.layers:
            x = layer(x)
            
        return x



class Discriminator(tf.keras.Model):
    def __init__(self, original_dim, filters_list, scales_list, initializer, 
                 num_features, image, adversial=False):
        
        super(Discriminator, self).__init__()
        
        self.conv_layers = []
        self.output_layers = []
        self.image = image
        self.original_dim = original_dim
        
        for filters, scales in zip(filters_list, scales_list):
            self.conv_layers.append(DiscriminatorScale(
                original_dim, filters, scales, initializer, num_features, image=image))
                    
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


'''
set_seed(seed=123)
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123)

# Create a generator model instance
latent_dim = 32
original_dim = 2048
batch_size  = 32
image = False
num_features = 4

discriminator_filters = [[64, 128, 256]]
discriminator_scales = [[2,2,2]]

generator_filters =  [[256, 128, 64], [64, 32, 32]]
generator_scales = [[2,1,1], [2,1,1]]
latent = False

generator = Generator(original_dim, generator_filters, generator_scales,
                      initializer, num_features, latent, image)

discriminator = Discriminator(original_dim, discriminator_filters, discriminator_scales,
                               initializer, num_features, image)

# Generate a sample input
#sample_input = tf.random.normal((batch_size, latent_dim))
sample_input = tf.random.normal((batch_size, original_dim, num_features))
#sample_input = tf.random.normal((batch_size, original_dim, original_dim, num_features))

#print('input ', sample_input.shape)

output = generator(sample_input)
#print('output ', output[0].shape, output[1].shape)
output_reshape = scale_fake_examples(output, original_dim, image)

#print('gen output')
#for out in output_reshape:
#    print(out.shape)
   

fake_concatenated = tf.concat((output_reshape[0], output_reshape[1]), axis=-1)

#print('dis_output')
dis_output = discriminator(fake_concatenated)
print(dis_output.shape)
'''