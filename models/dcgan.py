from tensorflow.keras import layers, Model
import tensorflow as tf 
from tensorflow_addons.layers import SpectralNormalization
from losses import discriminator_loss, generator_loss, mode_seeking_loss
from diff_aug import diff_augment

def build_generator(eeg_dim = 128, noise_dim = 100):

    eeg_input = layers.Input(shape = (eeg_dim, ), name = 'Embedding')
    noise_input = layers.Input(shape = (noise_dim, ), name = 'Noise')
    input_x = layers.Concatenate(name = 'Concat')([eeg_input, noise_input])

    x = layers.Dense(512, use_bias = False, name = 'Dense')(input_x)

    x = layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1), name = 'Reshape')(x)

    x = SpectralNormalization(layers.Conv2DTranspose(filters = 512, kernel_size = 3, strides = 4, padding = 'same', use_bias = False), name = 'Conv_1')(x)
    x = layers.BatchNormalization(name = 'Bnorm_1')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_1')(x)

    x = SpectralNormalization(layers.Conv2DTranspose(filters = 256, kernel_size = 3, strides = 2, padding = 'same', use_bias = False), name = 'Conv_2')(x)
    x = layers.BatchNormalization(name = 'Bnorm_2')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_2')(x)

    x = SpectralNormalization(layers.Conv2DTranspose(filters = 128, kernel_size = 3, strides = 2, padding = 'same', use_bias = False), name = 'Conv_3')(x)
    x = layers.BatchNormalization(name = 'Bnorm_3')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_3')(x)

    x = SpectralNormalization(layers.Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same', use_bias = False), name = 'Conv_4')(x)
    x = layers.BatchNormalization(name = 'Bnorm_4')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_4')(x)

    x = SpectralNormalization(layers.Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = 'same', use_bias = False), name = 'Conv_5')(x)
    x = layers.BatchNormalization(name = 'Bnorm_5')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_5')(x)

    x = SpectralNormalization(layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same', use_bias = False), name = 'Conv_6')(x)
    x = layers.BatchNormalization(name = 'Bnorm_6')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_6')(x)

    output = SpectralNormalization(layers.Conv2DTranspose(filters = 3, kernel_size = 3, strides = 1, padding = 'same', activation = 'tanh', use_bias = False), name = 'Output_Conv')(x)

    return Model([noise_input, eeg_input], output, name = 'Generator')

def build_discriminator(eeg_dim=128):
    input_image = layers.Input(shape=(128, 128, 3), name = 'Image')
    input_eeg = layers.Input(shape=(eeg_dim,), name = 'Embedding')

    # Image path
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', name = 'Conv_1')(input_image)
    x = layers.BatchNormalization(name = 'BNorm_1')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_1')(x)
    x = layers.Dropout(0.3, name = 'Dropout_1')(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', name = 'Conv_2')(x)
    x = layers.BatchNormalization(name = 'BNorm_2')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_2')(x)
    x = layers.Dropout(0.3, name = 'Dropout_2')(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', name = 'Conv_3')(x)
    x = layers.BatchNormalization(name = 'BNorm_3')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_3')(x)
    x = layers.Dropout(0.3, name = 'Dropout_3')(x)

    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same', name = 'Conv_4')(x)
    x = layers.BatchNormalization(name = 'BNorm_4')(x)
    x = layers.LeakyReLU(0.2, name = 'Activation_4')(x)
    x = layers.Dropout(0.3, name = 'Dropout_4')(x)

    x = layers.Flatten(name = 'Flat')(x)

    x = layers.Dense(128, name = 'Downsampling')(x)

    eeg_proj = layers.Dense(256, activation='relu', name = 'EEG_projection')(input_eeg)
    cond = eeg_proj

    merged = layers.Concatenate(name = 'Concat')([x, cond])
    merged = layers.Dense(256, name = 'Dense')(merged)
    merged = layers.LeakyReLU(0.2, name = "Activation_5")(merged)

    out = layers.Dense(1, name = 'Output_dense')(merged)  

    return Model([input_image, input_eeg], out, name="Discriminator")

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

        fake_diffused1 = diff_augment( fake1, policy = augment_policies)
        fake_diffused2 = diff_augment( fake2, policy = augment_policies)
        real_diffused  = diff_augment( real_img, policy = augment_policies)

        fake_out1 = dis([fake_diffused1, condition], training = True)
        fake_out2 = dis([fake_diffused2, condition], training = True)
        real_out  = dis([real_diffused, condition], training = True)

        d_loss = ( discriminator_loss(real_out, fake_out1) + discriminator_loss(real_out, fake_out2) ) / 2.0

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

    dis = build_discriminator()
    dis.build(input_shape = [(64, 64, 3), (128, )])
    dis.summary()
