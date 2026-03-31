from tensorflow.keras import layers, Model
import tensorflow as tf 
from tensorflow_addons.layers import SpectralNormalization
from losses import dis_loss, generator_loss, mode_seeking_loss
from diffusion import Diffusion

def build_generator(eeg_dim = 128, noise_dim = 100, output_res = 64):

    eeg_input = layers.Input(shape = (eeg_dim, ), name = 'Embedding')
    noise_input = layers.Input(shape = (noise_dim, ), name = 'Noise')
    input_x = layers.Concatenate(name = 'Concat')([eeg_input, noise_input])

    x = layers.Dense(512, use_bias = False, name = 'Dense')(input_x)

    x = layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1), name = 'Reshape')(x)

    x = layers.Conv2DTranspose(filters = 512, kernel_size = 3, strides = 2, padding = 'same', use_bias = False, name = 'Conv_1')(x)
    x = layers.BatchNormalization(name = 'Bnorm_1')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_1')(x)

    x = layers.Conv2DTranspose(filters = 256, kernel_size = 3, strides = 2, padding = 'same', use_bias = False, name = 'Conv_2')(x)
    x = layers.BatchNormalization(name = 'Bnorm_2')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_2')(x)

    x = layers.Conv2DTranspose(filters = 128, kernel_size = 3, strides = 2, padding = 'same', use_bias = False, name = 'Conv_3')(x)
    x = layers.BatchNormalization(name = 'Bnorm_3')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_3')(x)

    x = layers.Conv2DTranspose(filters = 128, kernel_size = 3, strides = 2, padding = 'same', use_bias = False, name = 'Conv_4')(x)
    x = layers.BatchNormalization(name = 'Bnorm_4')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_4')(x)

    x = layers.Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same', use_bias = False, name = 'Conv_5')(x)
    x = layers.BatchNormalization(name = 'Bnorm_5')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_5')(x)

    x = layers.Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same', use_bias = False, name = 'Conv_6')(x)
    x = layers.BatchNormalization(name = 'Bnorm_6')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_6')(x)

    output = layers.Conv2DTranspose(filters = 3, kernel_size = 3, strides = 1, padding = 'same', activation = 'tanh', use_bias = False, name = 'Output_Conv')(x)

    return Model([noise_input, eeg_input], output, name = 'Generator')

def build_dis(eeg_dim=128, timestep_dim=1):
    input_image = layers.Input(shape=(64, 64, 3), name = 'Image')
    input_eeg = layers.Input(shape=(eeg_dim,), name = 'Embedding')
    input_timestep = layers.Input(shape=(timestep_dim,), name = 'Timestep')

    # Image path
    x = SpectralNormalization(layers.Conv2D(64, kernel_size=4, strides=2, padding='same'), name = 'Conv_1')(input_image)
    x = layers.BatchNormalization(name = 'BNorm_1')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_1')(x)
    x = layers.Dropout(0.3, name = 'Dropout_1')(x)

    x = SpectralNormalization(layers.Conv2D(128, kernel_size=4, strides=2, padding='same'), name = 'Conv_2')(x)
    x = layers.BatchNormalization(name = 'BNorm_2')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_2')(x)
    x = layers.Dropout(0.3, name = 'Dropout_2')(x)

    x = SpectralNormalization(layers.Conv2D(256, kernel_size=4, strides=2, padding='same'), name = 'Conv_3')(x)
    x = layers.BatchNormalization(name = 'BNorm_3')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_3')(x)
    x = layers.Dropout(0.3, name = 'Dropout_3')(x)

    x = SpectralNormalization(layers.Conv2D(512, kernel_size=4, strides=2, padding='same'), name = 'Conv_4')(x)
    x = layers.BatchNormalization(name = 'BNorm_4')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_4')(x)
    x = layers.Dropout(0.3, name = 'Dropout_4')(x)

    x = layers.Flatten(name = 'Flat')(x)

    x = layers.Dense(256, name = 'Downsampling')(x)

    # EEG + timestep path
    eeg_proj = layers.Dense(256, activation='relu', name = 'EEG_projection')(input_eeg)
    time_proj = layers.Dense(64, activation='relu')(input_timestep)
    cond = layers.Concatenate()([eeg_proj, time_proj])  # shape: (320,)
    cond = eeg_proj

    # Merge image and condition features
    merged = layers.Concatenate(name = 'Concat')([x, cond])
    merged = layers.Dense(256, name = 'Dense')(merged)
    merged = layers.LeakyReLU(0.2, name = "Activation_5")(merged)

    out = layers.Dense(1, name = 'Output_dense')(merged)  # no activation → can use sigmoid or hinge loss externally

    # return Model([input_image, input_eeg, input_timestep], out, name="dis")
    return Model([input_image, input_eeg], out, name="dis")

bce = tf.keras.losses.BinaryCrossentropy(from_logits = True)

@tf.function
def train_step(gen, dis, g_opt, d_opt, condition, real_img, mode_scaling = 1.0, noise_dim = 100):

    augment_policies = 'color,translation'

    batch_size = tf.shape(condition)[0]
    noise1 = tf.random.normal((batch_size, noise_dim))
    noise2 = tf.random.normal((batch_size, noise_dim))
    t = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)

    with tf.GradientTape(persistent = True) as tape:

        fake1 = gen([noise1, condition], training = True)
        fake2 = gen([noise2, condition], training = True)

        diffusion_process = Diffusion(
            t_min=100,
            t_max=5000,
            beta_schedule='continuous_t',
            ts_dist='uniform', 
            # aug = 'diff',
            aug = 'no',
            beta_end = 0.1
        )

        fake_diffused1, t1 = diffusion_process(fake1, t)
        fake_diffused2, t2 = diffusion_process(fake2, t)
        real_diffused, tr  = diffusion_process(real_img, t)

        fake_out1 = dis([fake_diffused1, condition, t1], training = True)
        fake_out2 = dis([fake_diffused2, condition, t2], training = True)
        real_out  = dis([real_diffused, condition, tr], training = True)

        d_loss = ( dis_loss(real_out, fake_out1) + dis_loss(real_out, fake_out2) ) / 2.0

        g_loss = ( generator_loss(fake_out1) + generator_loss(fake_out2) ) / 2.0
        ms_loss = mode_seeking_loss(fake1, fake2, noise1, noise2)
        g_loss += mode_scaling * ms_loss

    d_gradients = tape.gradient(d_loss, dis.trainable_variables)
    g_gradients = tape.gradient(g_loss, gen.trainable_variables)

    d_opt.apply_gradients(zip(d_gradients, dis.trainable_variables))
    g_opt.apply_gradients(zip(g_gradients, gen.trainable_variables))

    return g_loss, d_loss

if __name__=='__main__': 
    
    gen = build_generator()
    gen.build(input_shape = [(100,), (128,)])
    gen.summary()

    dis = build_dis()
    dis.build(input_shape = [(64, 64, 3), (128, ), (1, )])
    dis.summary()
