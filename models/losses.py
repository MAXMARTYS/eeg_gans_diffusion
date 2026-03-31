import tensorflow as tf

def discriminator_loss(real_output, fake_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    return bce(tf.ones_like(fake_output), fake_output)

def mode_seeking_loss(fake1, fake2, noise1, noise2):
    epsilon = 1e-5
    img_diff = tf.reduce_mean(tf.abs(tf.subtract(fake1, fake2)))
    noise_diff = tf.reduce_mean(tf.abs(tf.subtract(noise1, noise2)))
    mode_loss = tf.divide(img_diff, noise_diff)
    mode_loss = tf.divide(1.0, mode_loss + epsilon)
    return mode_loss
