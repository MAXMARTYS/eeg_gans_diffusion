import tensorflow as tf 
import numpy as np
from scipy.optimize import linear_sum_assignment
import os 

def preprocess_data(X, Y):
	X = tf.squeeze(X, axis=-1)
	max_val = tf.reduce_max(X)/2.0
	X = (X - max_val) / max_val
	X = tf.transpose(X, [1, 0])
	X = tf.cast(X, dtype=tf.float32)
	Y = tf.argmax(Y) # NOTE: they take argmax --> label is an integer
	return X, Y

@tf.function
def parse(signal, objective):
    X, y = tf.py_function(preprocess_data, inp=[signal, objective], Tout=[tf.float32, tf.int64])
    X.set_shape((32, 14)) # NOTE: so far the shape is set manually, need to find a way to set it dynamically
    y.set_shape(())
    return X, y


# Triplet semi-hard loss
def pairwise_distance(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.math.equal(distances, 0.0)
        distances = tf.sqrt(distances + tf.cast(mask, tf.float32) * 1e-16)
        distances = distances * (1.0 - tf.cast(mask, tf.float32))

    return distances

def masked_maximum(data, mask, dim=1):
    axis_minimums = tf.reduce_min(data, axis=dim, keepdims=True)
    masked_maximums = tf.reduce_max(tf.multiply(data - axis_minimums, mask), axis=dim, keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    axis_maximums = tf.reduce_max(data, axis=dim, keepdims=True)
    masked_minimums = tf.reduce_min(tf.multiply(data - axis_maximums, mask), axis=dim, keepdims=True) + axis_maximums
    return masked_minimums

def custom_triplet_semihard_loss(y_true, y_pred, margin=1.0):
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")
    precise_embeddings = tf.cast(embeddings, tf.float32)

    labels = tf.reshape(labels, [-1, 1])
    pdist_matrix = pairwise_distance(precise_embeddings, squared=False)

    adjacency = tf.math.equal(labels, tf.transpose(labels))
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(tf.tile(adjacency_not, [batch_size, 1]), tf.math.greater(pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])))

    mask_final = tf.reshape(tf.math.greater(tf.reduce_sum(tf.cast(mask, tf.float32), axis=1, keepdims=True), 0.0), [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)

    negatives_outside = tf.reshape(masked_minimum(pdist_matrix_tile, tf.cast(mask, tf.float32)), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)
    negatives_inside = tf.tile(masked_maximum(pdist_matrix, tf.cast(adjacency_not, tf.float32)), [1, batch_size])

    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)
    loss_mat = tf.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, tf.float32) - tf.linalg.diag(tf.ones([batch_size]))
    num_positives = tf.reduce_sum(mask_positives)

    triplet_loss = tf.math.divide_no_nan(tf.reduce_sum(tf.maximum(tf.multiply(loss_mat, mask_positives), 0.0)), num_positives)

    return triplet_loss

# Accuracy figure
def cluster_acc(y_true, y_pred):
    """Compute clustering accuracy using the Hungarian algorithm."""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1  # Number of unique classes
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1  # Create confusion matrix

    row_ind, col_ind = linear_sum_assignment(w.max() - w)  # Solve assignment problem
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size  # Compute accuracy

def extract_embeddings(model, dataset):
    embeddings = []
    labels = []
    
    for X, y in dataset:
        Y_emb, _ = model(X, training=False)  # Forward pass with training=False
        embeddings.append(Y_emb.numpy())
        labels.append(y.numpy())
        
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels
    
def index_to_class(idx):
    return os.listdir(r'data\\ThoughtViz_data\\images\\ImageNet-Filtered')[idx]