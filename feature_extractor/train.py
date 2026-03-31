import pickle
import tensorflow as tf 
import tensorflow_addons as tfa
from extractor import TripleNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import parse, preprocess_data

if __name__=='__main__':

    path = r'data\data.pkl'
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']

    batch_size = 256

    # Datasets for training based on supercategory
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.map(parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    softnet = TripleNet(n_classes = 10, n_features = 128)
    softnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-4))

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 200,
        mode = 'min'
    )
    checkpoint = ModelCheckpoint(
        'checkpoint/best_triplenet.h5',
        monitor = "val_loss",
        save_best_only = True,
        save_weights_only = True
    )

    # Train the model
    epochs = 1000
    history = softnet.fit(
        train_dataset, 
        validation_data = test_dataset, 
        epochs = epochs,
        callbacks = [early_stopping, checkpoint]
        )
