import tensorflow as tf 
import numpy as np 
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def index_to_class(idx):
    return os.listdir(r'data\\ThoughtViz_data\\images\\ImageNet-Filtered')[idx]

def generate_images(gen, dataset):

    images = []
    for signals, y in dataset: 
        batch_size = tf.shape(signals)[0]
        noise = tf.random.uniform((batch_size, 100), minval = -1, maxval = 1)
        images_batch = gen([noise, signals])
        images.extend(images_batch.numpy())

    images = np.stack(images, axis = 0)
    return images

import gc

def inception_score_per_class(images, labels, splits=10):

    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    imdict = {idx: [] for idx in unique_classes}

    for i, img in enumerate(images):
        idx = labels[i]
        imdict[idx].append(img)

    total_mean = 0
    total_std = 0
    results = {idx: {'mean_is': 0, 'std_is': 0, 'quality': 0, 'diversity': 0} for idx in unique_classes}
    all_scores = []
    for idx, images in imdict.items():
        cname = index_to_class(idx)
        images = np.array(images)

        # Recreate the Inception model
        inception_model = InceptionV3(include_top=True, weights='imagenet')
        inception_model = Model(inception_model.input, inception_model.output)

        # Preprocess and create tf.data.Dataset to avoid memory issues
        images = preprocess_input(images.astype(np.float32))
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        # Run prediction
        preds = inception_model.predict(dataset, verbose=0)

        # Compute IS
        N = preds.shape[0]
        split_size = N // splits
        scores = []

        # ---- Quality (avg entropy of p(y|x)) ----
        avg_quality = np.mean([-np.sum(p * np.log(p + 1e-16)) for p in preds])

        # ---- Diversity (entropy of p(y)) ----
        marginal_py = np.mean(preds, axis=0)
        diversity = -np.sum(marginal_py * np.log(marginal_py + 1e-16))

        IS = np.exp(diversity - avg_quality)

        print(f"\nClass: {cname}")
        # print(f"  Inception Score       : {mean_is:.2f} ± {std_is:.2f}")
        print(f"  Inception Score       : {IS:.2f}")
        print(f"  Quality (avg entropy) : {avg_quality:.4f}")
        print(f"  Diversity (entropy)   : {diversity:.4f}")

        # total_mean += mean_is
        # total_std += std_is
        # print(f"Inception Score: {mean_is:.2f} ± {std_is:.2f}")

        # Free memory
        del inception_model
        tf.keras.backend.clear_session()
        gc.collect()
    
        # results[idx]['mean_is'] = mean_is
        # results[idx]['std_is'] = std_is
        results[idx]['quality'] = avg_quality
        results[idx]['diversity'] = diversity

        all_scores.extend(scores)

    # total_mean /= n_classes
    # total_std /= n_classes
    # print(f"\nInception Score Overall: {total_mean:.2f} ± {total_std:.2f}")

    return results, all_scores