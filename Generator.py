import tensorflow as tf
from PAD import PAD
tfkl = tf.keras.layers


class ConvLayer(tf.keras.Model):
    def __init__(self, original_dim, filters, scales, initializer):
        
        super(ConvLayer, self).__init__()

        self.conv_layers = []
        self.conv_layers.append(tfkl.UpSampling1D(size=scales))
            
        self.conv_layers.append(PAD())
        self.conv_layers.append(tfkl.Conv1D(filters, kernel_size=3, 
                                             strides=1, padding="valid",
                                               kernel_initializer=initializer,
                                               use_bias=False
                                               ))
            
        self.conv_layers.append(tfkl.BatchNormalization())
        self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))


    def call(self, inputs):
        x = inputs

        for layer in self.conv_layers.layers:
            x = layer(x)
                  
        return x


class GeneratorScale(tf.keras.Model):
    def __init__(self, original_dim, filters_list, scales_list, initializer):
        
        super(GeneratorScale, self).__init__()                
        self.conv_layers = []
        
        for filters, scales in zip(filters_list, scales_list):
            self.conv_layers.append(ConvLayer(original_dim, filters, scales, 
                                              initializer))          

    def call(self, inputs):
        x = inputs
 
        for layer in self.conv_layers.layers:
            x = layer(x)
            
        return x



class GeneratorOutput(tf.keras.Model):
    def __init__(self, initializer, num_features):
        
        super(GeneratorOutput, self).__init__()
        self.conv_layers = []

        # The 1 x 1 convolution for intermediate/final outputs
        self.conv_layers.append(PAD())
        self.conv_layers.append(tfkl.Conv1D(num_features, kernel_size=3, 
                                strides=1, padding="valid",
                                kernel_initializer=initializer,
                                use_bias=False
                                ))
            
        self.conv_layers.append(tfkl.BatchNormalization())
        self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))


    def call(self, inputs):
        x = inputs

        for layer in self.conv_layers.layers:
            x = layer(x)
                  
        return x



class Generator(tf.keras.Model):
    def __init__(self, original_dim, filters_list, scales_list, initializer, num_features,
                 latent=True):
        
        super(Generator, self).__init__()
                        
        
        # First three filters to reach the orginal dimensions
        require_dim = original_dim // 2**len(filters_list[0])
        self.flat = tfkl.Flatten()
        self.latent = latent
        
        if self.latent:
                        
            self.dense = tfkl.Dense(require_dim * filters_list[0][0], 
                                    kernel_initializer=initializer,
                                    use_bias=False)
                
            self.reshape = tfkl.Reshape((require_dim, filters_list[0][0]))
                
        self.conv_layers = []
        self.gen_output_layers = []
        
        for filters, scales in zip(filters_list, scales_list):
                        
            #each layer reprsent a stage to reach output at a certain resolution
            #can have multiple up-scaling, followed by 1x1 conv layer for output
            self.conv_layers.append(GeneratorScale(original_dim, filters, scales, initializer))
            self.gen_output_layers.append(GeneratorOutput(initializer, num_features))


    def call(self, inputs):
        x = inputs
        
        #sampling from Gaussian noise
        if self.latent:
            x = self.dense(self.flat(x))
            x = self.reshape(x)
        
        outputs = []
        for (layer, layer_output) in zip(self.conv_layers.layers, 
                                         self.gen_output_layers):
            x = layer(x)
            outputs.append(layer_output(x))
         
        return outputs


'''
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123)

# Create a generator model instance
latent_dim = 32
original_dim = 2048
batch_size  = 32
latent = True
num_features = 2

generator = Generator(original_dim, [[32, 64, 128], [128, 128, 128]],
                      [[2,2,2],[2,1,1]],
                      initializer, num_features, latent)

# Generate a sample input
sample_input = tf.random.normal((batch_size, original_dim, num_features))
#sample_input = tf.random.normal((batch_size, original_dim, original_dim, num_features))

print('input ', sample_input.shape)
output = generator(sample_input)
print('output ')
for out in output:
    print(out.shape)
'''
