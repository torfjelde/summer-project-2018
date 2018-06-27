#!/usr/bin/env python
import numpy as np

from tqdm import tqdm

import logging

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)-15s %(levelname)-9s %(module)s: %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL))

log = logging.getLogger("rbm")


def loss(samples_true, samples):
    "Computes the difference in empirical distributions."
    return np.mean(np.abs(np.mean(samples_true, axis=0) - np.mean(samples, axis=0)))


def sigmoid(z):
    # clip the values due to possibility of overflow
    return 1.0 / (1.0 + np.exp(-np.maximum(np.minimum(z, 30), -30)))


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
        W = np.random.normal(0.0, 0.01, (num_visible, num_hidden))
        
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
    
    def contrastive_divergence(self, v_0, k=1):
        h = self.sample_hidden(v_0)
        v = self.sample_visible(h)
        
        if k > 1:
            for t in range(k):
                h = self.sample_hidden(v)
                v = self.sample_visible(h)
                
        return v_0, v

    def grad(self, v, k=1):
        "Estimates the gradient of the negative log-likelihood using CD-k."
        v_0, v_k = self.contrastive_divergence(v, k=k)
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
    
    def step(self, vs, k=1, lr=0.1):
        "Performs a single gradient descent step using CD-k on the batch `vs`."
        # TODO: can we perform this over the batch using matrix multiplication instead?
        delta_c, delta_b, delta_W = self.grad(vs[0], k=k)
        for v in vs[1:]:
            # compute gradient for each observed visible configuration
            dc, db, dW = self.grad(v, k=k)
            # accumulate gradients
            delta_c += dc
            delta_b += db
            delta_W += dW

        # update parameters
        rbm.c += lr * (delta_c / len(vs))
        rbm.b += lr * (delta_b / len(vs))
        rbm.W += lr * (delta_W / len(vs))
        
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
        samples = []
        v = samples_true[0]

        for i in range(len(samples_true)):
        #     v = history[i]
            h = self.sample_hidden(v)
            p = self.proba_visible(h)
            
            for j in range(per_sample_hidden):
                p += self.proba_visible(self.sample_hidden(v))
                
            p /= per_sample_hidden
            samples.append(v)
            
        return loss(samples_true, samples)


if __name__ == "__main__":
    NUM_EPOCHS = 1
    BATCH_SIZE = 128
    NUM_HIDDEN = 250
    LEARNING_RATE = 0.01
    K = 1

    import os
    from six.moves import urllib
    from sklearn.datasets import fetch_mldata


    # Alternative method to load MNIST, since mldata.org is often down...
    from scipy.io import loadmat
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"

    if os.path.exists(mnist_path):
        log.info(f"Found existing file at {mnist_path}")
        mnist_raw = loadmat(mnist_path)
        mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
        }
    else:
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

    X = mnist["data"]
    y = mnist["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train / np.linalg.norm(X_train, axis=1).reshape(-1, 1)
    X_test = X_test / np.linalg.norm(X_test, axis=1).reshape(-1, 1)

    # model
    rbm = BernoulliRBM(X_train.shape[1], NUM_HIDDEN)

    # train
    log.info(f"Training")
    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for epoch in range(1, NUM_EPOCHS + 1):
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
    
    num_samples = 1000

    fig, axes = plt.subplots(5, 2, figsize=(14, 4))
    for i in range(5):
        x = X_test[np.random.randint(X_test.shape[0])]
        probs = rbm.sample_visible(rbm.sample_hidden(x))

        for _ in range(num_samples):
            probs += rbm.sample_visible(rbm.sample_hidden(x))

        probs = probs / num_samples

        axes[i][0].imshow(np.reshape(probs, (28, 28)))
        axes[i][1].imshow(np.reshape(x, (28, 28)))
    plt.savefig("sample.png")
    # plt.show()

