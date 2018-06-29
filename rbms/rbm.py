#!/usr/bin/env python
import argparse

# progess-bar
from tqdm import tqdm

import logging

### Parsing ###
parser = argparse.ArgumentParser(description="Trains an RBM on the MNIST dataset.")
parser.add_argument("-k", type=int, default=1, help="Number of steps to use in Contrastive Divergence (CD-k)")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--hidden-size", type=int, default=500, help="Number of hidden units")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs; one epoch runs through entire training data")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate used multiplied by the gradients")
parser.add_argument("--gpu", action="store_true", help="Whether or not to use the GPU. Requires CUDA and cupy installed.")
parser.add_argument("--output", type=str, default="sample.png", help="Output file for reconstructed images from test data")
parser.add_argument("--show", type=bool, default=False, help="Whether or not to display image; useful when running on remote computer")
parser.add_argument("-L", "--loglevel", type=str, default="INFO")

FLAGS = parser.parse_args()

### Numpy or Cupy?
if FLAGS.gpu:
    import cupy as np
else:
    import numpy as np

### Logging ###
LOG_LEVEL = FLAGS.loglevel
LOG_FORMAT = '%(asctime)-15s %(levelname)-9s %(module)s: %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL))
log = logging.getLogger("rbm")


### Utilities
def loss(samples_true, samples):
    "Computes the difference in empirical distributions."
    return np.mean(np.abs(np.mean(samples_true, axis=0) - np.mean(samples, axis=0)))


def sigmoid(z):
    # clip the values due to possibility of overflow
    return 1.0 / (1.0 + np.exp(-np.maximum(np.minimum(z, 30), -30)))


### Restricted Boltzmann Machine ###
class BernoulliRBM(object):
    """
    RBM with Bernoulli variables for hidden and visible states.
    """
    def __init__(self, num_visible, num_hidden):
        super(BernoulliRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.c, self.b, self.W = self.initialize(self.num_visible, self.num_hidden)
      
    @staticmethod
    def initialize(num_visible, num_hidden):
        # biases for visible and hidden, respectively
        c = np.zeros(num_visible)
        b = np.zeros(num_hidden)

        # weight matrix
        # Results on MNIST are highly dependent on this initialization
        W = np.random.normal(0.0, 1.0, (num_visible, num_hidden))
        
        return c, b, W

    def energy(self, v, h):
        return - (np.dot(self.c, v) + np.dot(self.b, h) + np.dot(v, np.dot(self.W, h)))

    def proba_visible(self, h):
        "Computes p(v | h)."
        return sigmoid(self.c + np.dot(self.W, h))

    def proba_hidden(self, v):
        "Computes p(h | h)."
        return sigmoid(self.b + np.dot(self.W.T, v))

    def sample_visible(self, h):
        "Samples visible units from the given hidden units `h`."
        # compute p(V_j = 1 | h)
        probas = self.proba_visible(h)
        # equiv. of V_j ~ p(V_j | h)
        rands = np.random.random(size=probas.shape)
        v = (probas > rands).astype(int)
        return v

    def sample_hidden(self, v):
        "Samples hidden units from the given visible units `v`."
        # compute p(H_{\mu} = 1 | v)
        probas = self.proba_hidden(v)
        # euqiv. of H_{\mu} ~ p(H_{\mu} | h)
        rands = np.random.random(size=probas.shape)
        h = (probas > rands).astype(np.int)
        return h

    def free_energy(self, v):
        # unnormalized
        # F(v) = - log \tilde{p}(v) = - \log \sum_{h} \exp ( - E(v, h))
        # using Eq. 2.20 (Fischer, 2015) for \tilde{p}(v)
        visible = np.dot(self.c, v)
        hidden = self.b + np.dot(self.W, v)
        return - (visible + np.sum(np.log(1 + np.exp(hidden))))
    
    def contrastive_divergence(self, v_0, k=1):
        h = self.sample_hidden(v_0)
        v = self.sample_visible(h)
        
        if k > 1:
            for t in range(k):
                h = self.sample_hidden(v)
                v = self.sample_visible(h)
                
        return v_0, v

    def grad(self, v_0, v_k, k=1):
        "Estimates the gradient of the negative log-likelihood using CD-k."
        proba_h_0 = self.proba_hidden(v_0)
        proba_h_k = self.proba_hidden(v_k)
        
        delta_c = v_0 - v_k
        delta_b = proba_h_0 - proba_h_k

        # reshape so that we can compute v_j h_{\mu} by
        # taking the dot product to obtain `delta_W`
        v_0 = np.reshape(v_0, (-1, 1))
        proba_h_0 = np.reshape(proba_h_0, (1, -1))
        
        v_k = np.reshape(v_k, (-1, 1))
        proba_h_k = np.reshape(proba_h_k, (1, -1))
        
        delta_W = np.dot(v_0, proba_h_0) - np.dot(v_k, proba_h_k)
        
        return delta_c, delta_b, delta_W
    
    def step(self, vs, k=1, lr=0.1, lmda=0.0):
        "Performs a single gradient ascent step using CD-k on the batch `vs`."
        # TODO: can we perform this over the batch using matrix multiplication instead?
        v_0, v_k = self.contrastive_divergence(vs[0], k=k)
        delta_c, delta_b, delta_W = self.grad(v_0, v_k, k=k)
        for v in vs[1:]:
            # perform CD-k
            v_0, v_k = self.contrastive_divergence(v, k=k)
            # compute gradient for each observed visible configuration
            dc, db, dW = self.grad(v_0, v_k, k=k)
            # accumulate gradients
            delta_c += dc
            delta_b += db
            delta_W += dW

        # update parameters
        self.c += lr * (delta_c / len(vs))
        self.b += lr * (delta_b / len(vs))
        self.W += lr * (delta_W / len(vs))

        # possible apply weight-decay
        if lmda > 0.0:
            self.c -= lmda * self.c
            self.b -= lmda * self.b
            self.W -= lmda * self.W
        
    def loss(self, samples_true, per_sample_hidden=100):
        """
        Computes the difference in empirical distributions observed in `samples_true` and samples
        of visible units obtained from the model, using the same initial state.

        Parameters
        ----------
        samples_true: array-like
            True samples of visible units to compare to.
        per_sample_hidden: int
            Number of `h` to "partially marginalize" over when computing p(v) = p(v | h) p(h).

        Returns
        -------
        loss: float
            Difference between the two distributions.

        Notes
        -----
        Can be very inaccurate estimate of how well the model is performing since one is NOT completely
        marginalizing out all hidden variables.

        """
        k = per_sample_hidden
        # the loss is the log energy-difference between the p(v) and p(v_k), where `v_k` is the Gibbs sampled visible unit
        return np.mean([
            self.free_energy(v) - self.free_energy(self.contrastive_divergence(v, k))
            for v in samples_true
        ])
        # samples = []
        # v = samples_true[0]

        # for i in range(len(samples_true)):
        # #     v = history[i]
        #     h = self.sample_hidden(v)
        #     p = self.proba_visible(h)
            
        #     for j in range(per_sample_hidden):
        #         p += self.proba_visible(self.sample_hidden(v))
                
        #     p /= per_sample_hidden
        #     samples.append(v)
            
        # return loss(samples_true, samples)

    def reconstruct(self, v, num_samples=100):
        samples = self.sample_visible(self.sample_hidden(v))
        for _ in range(num_samples - 1):
            samples += self.sample_visible(self.sample_hidden(v))

        probs = samples / num_samples 

        return probs


if __name__ == "__main__":
    NUM_EPOCHS = FLAGS.epochs
    BATCH_SIZE = FLAGS.batch_size
    NUM_HIDDEN = FLAGS.hidden_size
    LEARNING_RATE = FLAGS.lr
    K = FLAGS.k

    import os
    from six.moves import urllib
    from sklearn.datasets import fetch_mldata

    # Alternative method to load MNIST, since mldata.org is often down...
    from scipy.io import loadmat
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"

    if os.path.exists(mnist_path):
        log.info(f"Found existing file at {mnist_path}; loading...")
        mnist_raw = loadmat(mnist_path)
        mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
        }
    else:
        log.info(f"Dataset not found at {mnist_path}; downloading...")
        response = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            content = response.read()
            f.write(content)
        mnist_raw = loadmat(mnist_path)
        mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
        }
        log.info("Success!")

    # train-test split
    from sklearn.model_selection import train_test_split

    # in case we want to use `cupy` to run on the GPU
    X = np.asarray(mnist["data"])
    y = np.asarray(mnist["target"])

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # clip values since we're working with binary variables and original images have domain [0, 255]
    X_train = X_train.clip(0, 1)
    X_test = X_test.clip(0, 1)

    # model
    rbm = BernoulliRBM(X_train.shape[1], NUM_HIDDEN)

    # train
    log.info(f"Starting training")
    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # if epoch == NUM_EPOCHS:
        #     log.info(f"[{epoch} / {NUM_EPOCHS}] Increasing k: {K} -> {5 * K}")
        #     K = 5 * K
        # else:
        log.info(f"[{epoch} / {NUM_EPOCHS}]")

        bar = tqdm(total=num_samples)
        for start in range(0, num_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_samples)
            rbm.step(X_train[start: end], k=K, lr=LEARNING_RATE)
            bar.update(end - start)

        # shuffle indices for next epoch
        np.random.shuffle(indices)

    # display a couple of testing examples
    import matplotlib
    # don't use `Xwindows` to render since this doesn't necessarily work over SSH
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # plot some reconstructions
    from matplotlib import gridspec

    n_rows = 6
    n_cols = 8

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, 
                             figsize=(16, 12), 
                             gridspec_kw=dict(wspace=-0.1, hspace=-0.01))

    for i in range(n_rows):
        for j in range(n_cols // 2):
            v = X_test[np.random.randint(X_test.shape[0])]
            probs = rbm.reconstruct(v)

            # in case we've substituted with `cupy`
            if np.__name__ != "numpy":
                v = np.asnumpy(v)
                probs = np.asnumpy(probs)

            axes[i][2 * j].imshow(np.reshape(v, (28, 28)))
            axes[i][2 * j + 1].imshow(np.reshape(probs, (28, 28)))

            # customization; remove labels
            axes[i][2 * j].set_xticklabels([])
            axes[i][2 * j].set_yticklabels([])

            axes[i][2 * j + 1].set_xticklabels([])
            axes[i][2 * j + 1].set_yticklabels([])

    fig.savefig("test.png")
    log.info(f"Saving to {FLAGS.output}")
    plt.savefig(FLAGS.output)

    if FLAGS.show:
        plt.show()

