import tensorflow as tf 
import tensorflow_addons as tfa
from utils import custom_triplet_semihard_loss

# Extractor architecture taken from https://github.com/prajwalsingh/EEG2Image

class TripleNet(tf.keras.Model):

    def __init__(self, n_classes=10, n_features=128):
        super(TripleNet, self).__init__()
        filters = [32, n_features]
        ret_seq = [True, False]

        self.encoder = [tf.keras.layers.LSTM(units=filters[i], return_sequences=ret_seq[i])
                        for i in range(len(filters))]

        self.flat = tf.keras.layers.Flatten()
        self.w_1  = tf.keras.layers.Dense(units=n_features, activation='leaky_relu')
        self.w_2  = tf.keras.layers.Dense(units=n_features)

    def call(self, x):
        for layer in self.encoder:
            x = layer(x)

        x = feat = self.flat(x)
        x = self.w_2(self.w_1(x))
        x = tf.nn.l2_normalize(x, axis=-1)

        return x, feat

    def train_step(self, batch):
        X, y = batch 
        with tf.GradientTape() as tape:
            Y_emb, _ = self(X, training=True)  # Forward pass
            # Use custom_triplet_semihard_loss if tensorflow addons is not available
            loss  = tfa.losses.TripletSemiHardLoss(margin=0.2)(y, Y_emb)
             
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return {"loss": loss}

    def test_step(self, batch):
        X, y = batch 
        Y_emb, _ = self(X, training=False)  # Forward pass
        loss = tfa.losses.TripletSemiHardLoss(margin=0.2)(y, Y_emb)
        return {"loss": loss}
