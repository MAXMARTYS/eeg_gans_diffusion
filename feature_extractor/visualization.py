from extractor import TripleNet
from utils import extract_embeddings, cluster_acc, index_to_class
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf 
from utils import parse, preprocess_data
import pickle

if __name__=='__main__':

    path = r'data\data.pkl'
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']

    batch_size=256

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.map(parse, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    softnet = TripleNet(n_classes = 10, n_features = 128)
    softnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-4))

    dummy_input = tf.random.normal((1, 32, 14))  # Adjust based on your actual input shape
    _ = softnet(dummy_input, training=False)  


    # Load best model weights
    softnet.load_weights('models/best_triplenet_thoughtviz.h5')

    # Get embeddings for the test set
    embeddings, labels = extract_embeddings(softnet, test_dataset)

    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    kmeans_acc = cluster_acc(labels, cluster_labels)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # Plot with Matplotlib
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=index_to_class(label),
            color=colors[i],
            edgecolors='k',
            alpha=1.0
        )
    # plt.colorbar(scatter, label="Class Label")
    plt.legend(title="Class", loc='best', fontsize=10)
    plt.title(f"t-SNE Visualization of TripleNet Embeddings\nk-Means Clustering Accuracy = {kmeans_acc:.4f}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig('figures\\007-triplenet_thoughtviz_tsne.png')
    plt.grid(True)
    plt.show()
