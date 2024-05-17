import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numpy import random
import os

###############################################################################

def set_seed(seed=123):
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)

    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
##########################Loss function####################################


# reduce difference between scores and real and fake field
@tf.function
def discriminator_loss(real_logits, fake_logits):
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


# maximize the expectation of the critic's output on fake field
# opposite of discriminator score
@tf.function
def generator_loss(fake_logits_list):
    
    avg_g_loss = 0.0
    for fake in fake_logits_list:
        avg_g_loss += -tf.reduce_mean(fake)
    
    return avg_g_loss/tf.cast(len(fake_logits_list), dtype=tf.float32)



@tf.function(reduce_retracing=True)
def gradient_penalty(discriminator, real, fake):
        
    local_batch_size = tf.shape(real)[0]

    # Get the interpolated  1D data
    alpha = tf.random.uniform([local_batch_size, 1, 1], 0, 1)

    interpolated = real + alpha * (fake - real)
        
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
                        
        # 1. Get the discriminator output for this interpolated field.
        pred = discriminator(interpolated, training=True)
                    
    # 2. Calculate the gradients w.r.t to this interpolated field.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients all spatial dims, excluding batch
    axes_to_mean = tuple(range(1, len(real.shape)))
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axes_to_mean))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    
    
    return gp


# PDF loss function 
@tf.function
def pdf_loss_fn(real_list, fake_list):
        
    avg_pdf_loss = 0.0
    
    for real, fake in zip(real_list, fake_list):
        
        _, real_pdf = get_pdf(real)
        _, fake_pdf = get_pdf(fake)
            
        abs_diff = tf.math.abs(real_pdf - fake_pdf)
        
        # Mask bins where diff is equal to zero
        mask = tf.not_equal(abs_diff, 0)
        avg_pdf_loss += tf.reduce_mean(tf.boolean_mask(abs_diff, mask))

    return  avg_pdf_loss/tf.cast(len(fake_list), dtype=tf.float32)


@tf.function
def ps_loss_fn(real_list, fake_list, box_size_list):
        
    avg_ps_loss = 0.0
    for index, (real, fake) in enumerate(zip(real_list, fake_list)):

        real_k, real_ps = get_ps_fft(real, box_size_list[index])
        fake_k, fake_ps = get_ps_fft(fake, box_size_list[index])
            
        avg_ps_loss += tf.reduce_mean(tf.math.abs(real_ps - fake_ps))
        
    avg_ps_loss /= tf.cast(len(fake_list), dtype=tf.float32)
    
    return  avg_ps_loss

##########################PDF and PS routines####################################


@tf.function
def get_ps_fft(data, skewer_length=20):
        
    pi = tf.constant(np.pi, dtype=tf.float32)
    numlos = tf.shape(data)[0]
    nbins = tf.shape(data)[1]
    nfields = tf.shape(data)[-1]  # Get the number of fields
        
    # Create a TensorArray to store ps for each field
    ps_ta = tf.TensorArray(tf.float32, size=nfields)
    k_ta = tf.TensorArray(tf.float32, size=nfields)
    
    for field_idx in tf.range(nfields):
        field_data = data[...,field_idx]  # Select data for the current field
                
        ps = tf.zeros((nbins,), dtype=tf.float32)  # Initialize ps with the correct shape

        for i in tf.range(numlos):
            ps += tf.abs(tf.signal.fftshift(
                tf.signal.fft(tf.cast(field_data[i, :], tf.complex64))
                )) ** 2

        k_array = tf.range(-(nbins // 2), nbins // 2, dtype=tf.float32) /\
            tf.cast(skewer_length, dtype=tf.float32)

        numlos_float = tf.cast(numlos, tf.float32)
        k = 2 * pi * k_array
        ps *= skewer_length / numlos_float
        ps *= k / pi

        if nbins % 2 == 0:
            k = k[-(tf.cast(nbins/2, dtype=tf.int32))+1:]
            ps = ps[-(tf.cast(nbins/2, dtype=tf.int32))+1:]
        else:
            k = k[-(tf.cast(nbins/2, dtype=tf.int32)-1)+1:]
            ps = ps[-(tf.cast(nbins/2, dtype=tf.int32)-1)+1:]
            
        kmin = tf.reduce_min(k)
        kmax = tf.reduce_max(k)
        
        k_interp = tf.cast(tf.linspace(
            tf.experimental.numpy.log10(kmin), 
            tf.experimental.numpy.log10(kmax), 100), 
            dtype=tf.float32)

        # Interpolate power spectrum to desired k values
        ps_interp = tf.experimental.numpy.log10(
            tfp.math.interp_regular_1d_grid(
            x=tf.math.pow(10., k_interp),
            x_ref_min=kmin, 
            x_ref_max=kmax, 
            y_ref=ps,
        ))

        ps_ta = ps_ta.write(field_idx, ps_interp)
        k_ta = k_ta.write(field_idx, k_interp)
                                
    return k_ta.stack(), ps_ta.stack()  # Return k values and power spectra for all fields





@tf.function
def get_pdf(data):
    
    nfields = tf.shape(data)[-1]  # Get the number of fields
    nbins = 2000
    minimum_value = -10
    maximum_value =  10
    
    x_values_ta = tf.TensorArray(tf.float32, size=nfields)  # Create a TensorArray to store ps for each field
    pdf_ta = tf.TensorArray(tf.float32, size=nfields)  # Create a TensorArray to store ps for each field
    
    for field_idx in tf.range(nfields):
        
        # Select data for the current field and flatten
        field_data = tf.reshape(data[...,field_idx], (-1,))

        x_values = tf.range(minimum_value, maximum_value, dtype=tf.float32)
        
        x_bins_size = (maximum_value - minimum_value) / (nbins-1)
        x_values = tf.cast(tf.linspace(
            minimum_value, maximum_value, nbins),
            dtype=tf.float32)
        

        hist = tf.cast( tf.histogram_fixed_width(
            field_data, [minimum_value, maximum_value],
            nbins=nbins), dtype=tf.float32 )
        
        hist_sum = tf.reduce_sum(hist)
        
        # Normalize histogram
        if hist_sum>0.0:
            hist_normalized = hist/hist_sum/tf.cast(x_bins_size, dtype=tf.float32)
        else:
            hist_normalized = hist

        x_values_ta = x_values_ta.write(field_idx, x_values) 
        pdf_ta = pdf_ta.write(field_idx, hist_normalized)

            
    # Return k values and power spectra for all fields
    return x_values_ta.stack(), pdf_ta.stack()  


##########################Reshape routines####################################


@tf.function
def scale_fake_examples(fake_list, original_dim, batch_size):
    
    fake_reshaped_list = []
    
    for fake in fake_list:
        
        batch_size = tf.shape(fake)[0]
        fake_dim = tf.shape(fake)[1]
        num_features = tf.shape(fake)[-1]
        
        factor = fake_dim // original_dim
        
        new_dim = fake_dim // factor
        fake_reshape = tf.reshape(fake, (batch_size * factor, new_dim, num_features))
              
        indices = tf.random.shuffle(tf.range(tf.shape(fake_reshape)[0]))[:batch_size]
        fake_sample = tf.gather(fake_reshape, indices)    
        fake_reshaped_list.append(fake_sample)

    return fake_reshaped_list


@tf.function
def concat_as_channels(data_list):
    return tf.concat([x for x in data_list], axis=-1)


# Hubble function
@tf.function
def HubbleZ(z, omegam=0.305147, hubble=0.676):

    hubble = tf.constant(hubble)  # Assuming hubble constant value
    return hubble * 100. * tf.sqrt(omegam * (1. + z)**3 + (1. - omegam))
