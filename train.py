import tensorflow as tf 
import os 
from models.dcgan import build_discriminator, build_generator, train_step
# from models.diffgan import build_discriminator, build_generator, train_step
from tqdm import tqdm
import pickle
from feature_extractor.utils import preprocess_data, parse

checkpoint_dir = "./checkpoints/dcgan_experiments"
os.makedirs(checkpoint_dir, exist_ok=True)

generator = build_generator()
discriminator = build_discriminator()

g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Track models and optimizers
checkpoint = tf.train.Checkpoint(
    generator = generator,
    discriminator = discriminator,
    g_optimizer = g_optimizer,
    d_optimizer = d_optimizer
)

# Save the latest N checkpoints
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=50
)

checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    print(f"Restored from {checkpoint_manager.latest_checkpoint}")
else:
    print("Starting from scratch.")


def train(dataset, epochs):
    for epoch in range(epochs):
        # print(f"\poch {epoch+1}/{epochs}")
        
        progbar = tqdm.tqdm(
            dataset,
            desc=f"Epoch {epoch+1}",
            total=tf.data.experimental.cardinality(dataset).numpy(),
            dynamic_ncols=True,
            leave=True,
            unit="batch"
        )

        # Track total loss for averaging
        total_d_loss = 0.0
        total_g_loss = 0.0
        batch_count = 0

        for step, (eeg, real_img) in enumerate(progbar):
            # d_loss, g_loss = train_step(eeg, real_img)
            g_loss, d_loss = train_step(generator, discriminator, g_optimizer, d_optimizer, eeg, real_img, mode_scaling = 0.5)
            total_d_loss += d_loss.numpy()
            total_g_loss += g_loss.numpy()
            batch_count += 1
            progbar.set_postfix({"D loss": d_loss.numpy(), "G loss": g_loss.numpy()})

        # Compute average losses
        avg_d_loss = total_d_loss / batch_count
        avg_g_loss = total_g_loss / batch_count
        print(f"Epoch {epoch+1} | Avg D loss: {avg_d_loss:.4f}, Avg G loss: {avg_g_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            save_path = checkpoint_manager.save()
            print(f"Checkpoint saved at {save_path}")


if __name__=='__main__':

    path = r'feature_extractor\data\data.pkl'
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

    epochs = 100
    train(train_dataset, epochs)