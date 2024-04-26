import tensorflow as tf
from utility_functions import scale_fake_examples, gradient_penalty, concat_as_channels
tfkl = tf.keras.layers

###############################################################################
#WGAN CLASS for 1D data and 2D images
###############################################################################

class MSWGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        box_sizes,
        original_dim,
        latent,
        image=False,
        gp_weight=10.0,
    ):
        super(MSWGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.box_sizes = box_sizes
        self.original_dim = original_dim
        self.latent = latent
        self.image = image
        self.gp_weight = gp_weight


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, 
                g_loss_fn, ps_loss_fn, pdf_loss_fn):
        
        super(MSWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.ps_loss_fn = ps_loss_fn
        self.pdf_loss_fn = pdf_loss_fn
        
                
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.p_loss_metric = tf.keras.metrics.Mean(name="p_loss")
        self.pdf_loss_metric = tf.keras.metrics.Mean(name="pdf_loss")
        self.total_g_loss_metric = tf.keras.metrics.Mean(name="total_g_loss")


    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.p_loss_metric]
    
    
    #@tf.function
    def train_step(self, real_list):
        
        batch_size = tf.shape(real_list[0])[0]
    
        if self.latent:
            real = real_list
            box_sizes = self.box_sizes
        else:
            real = real_list[1:]
            box_sizes  = self.box_sizes[1:]
        
        noise = tf.random.normal(shape=(batch_size, self.latent_dim)) \
            if self.latent else real_list[0]
        
        # all dis/gen pairs trained together with in scope of gradient tape
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                                                
            fake = scale_fake_examples(self.generator(noise, training=True), 
                                       self.original_dim, batch_size, self.image)
            
            fake_concatenated = concat_as_channels(fake)
            real_concatenated = concat_as_channels(real)
            
            fake_logits = self.discriminator(fake_concatenated, training=True)
            real_logits = self.discriminator(real_concatenated, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_logits, fake_logits)
            
            # Calculate the gradient penalty
            gp = gradient_penalty(self.discriminator, real_concatenated, 
                                  fake_concatenated, self.image)
            g_loss = self.g_loss_fn(fake_logits)
            p_loss = self.ps_loss_fn(real, fake, box_sizes, self.image)
            pdf_loss = self.pdf_loss_fn(real, fake, self.image)
            d_loss =  d_cost + gp * self.gp_weight
            total_g_loss = g_loss + p_loss + pdf_loss
           

        # Get the gradients w.r.t the discriminator loss
        d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Get the gradients w.r.t the generator loss
        g_gradient = g_tape.gradient(total_g_loss, self.generator.trainable_variables)

         
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables))

        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.p_loss_metric.update_state(p_loss)
        self.pdf_loss_metric.update_state(pdf_loss)
        self.total_g_loss_metric.update_state(total_g_loss)
   
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "p_loss": self.p_loss_metric.result(),
            "pdf_loss": self.pdf_loss_metric.result(),
            "total_g_loss": self.total_g_loss_metric.result(),
            "dis. lr": self.d_optimizer.lr(self.d_optimizer.iterations)
        }

