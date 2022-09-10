import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os


class VI():
    def __init__(self, a0, b0, mu0, lambda0, Etao, threshold):
        self.a0 = a0
        self.b0 = b0
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.Etao = Etao
        self.muN = 0
        self.lambdaN = 0
        self.aN = 0
        self.bN = 0
        self.threshold = threshold

    def fit(self, X):
        print("\tDoing Variational Inference....")
        while True:
            old_Etao = self.Etao
            self.q_mu_update(X)
            self.q_tao_update(X)
            self.Etao = self.aN/self.bN
            if abs(self.Etao - old_Etao) < self.threshold:
                break

    def q_mu_update(self, X):
        mean, N = np.mean(X), X.shape[0]

        self.muN = ((self.lambda0*self.mu0) + N*mean) / (self.lambda0 + N)
        self.lambdaN = (self.lambda0 + N)*self.Etao

    def q_tao_update(self, X):
        mean, N = np.mean(X), X.shape[0]

        self.aN = self.a0 + N/2

        self.bN = np.dot(X, X) + (N + self.lambda0) * \
            (1/self.lambdaN + math.pow(self.muN, 2))
        self.bN = self.bN - 2*self.muN*(N*mean + self.mu0*self.lambda0)
        self.bN = self.bN + self.lambda0*math.pow(self.mu0, 2)
        self.bN = self.b0 + 0.5*self.bN

    def get_posterior(self, mu_axis, tao_axis):
        mu_size, tao_size = len(mu_axis), len(tao_axis)
        q = np.zeros(shape=(mu_size, tao_size))

        print('\tCalculating VI posterior approximation...')
        for i in range(mu_size):
            for j in range(tao_size):
                q_tao = stats.gamma.pdf(
                    tao_axis[j], self.aN, loc=0, scale=1/self.bN)
                q_mu = stats.norm.pdf(
                    mu_axis[i], self.muN, np.sqrt(1/self.lambdaN))
                q[j][i] = q_tao*q_mu
        return q


def generate_data(mu, tao, N, mu_axis, tao_axis, lambda0, mu0, a0, b0):
    sigma = np.sqrt(1/tao)
    X = np.random.normal(mu, sigma, N)
    true_posterior = get_true_posterior(
        mu_axis, tao_axis, X, lambda0, mu0, a0, b0)
    return X, true_posterior


def get_true_posterior(mu_axis, tao_axis, X, lambda0, mu0, a0, b0):
    mu_size, tao_size = len(mu_axis), len(tao_axis)
    p = np.zeros(shape=(mu_size, tao_size))
    mean, N = np.mean(X), X.shape[0]

    mu = (N*mean + lambda0*mu0)/(N + lambda0)
    a = a0 + N/2
    b = b0 + 0.5 * (np.dot(X, X) + lambda0*mu0**2 -
                    (N*mean + lambda0*mu0)**2/(N+lambda0))

    print('\tCalculating true posterior...')
    for i in range(mu_size):
        for j in range(tao_size):
            tao = (tao_axis[j]*(N+lambda0)) + sys.float_info.epsilon
            sigma = np.sqrt(1/tao)

            p_tao = stats.gamma.pdf(tao_axis[j], a, loc=0, scale=1/b)
            p_mu = stats.norm.pdf(mu_axis[i], mu, sigma)
            p[j][i] = p_tao*p_mu

    return p


def plot_contours(mu_axis, tao_axis, VI_posterior, true_posterior, filename, title):
    filename = f'2_3/plots/{filename}'

    plt.figure()
    CS_inferred = plt.contour(mu_axis, tao_axis, VI_posterior, colors='green')
    CS_true = plt.contour(mu_axis, tao_axis, true_posterior, colors='red')

    # Plot density of pdf on the contour line
    # plt.clabel(CS_inferred, fmt="%1.2f")
    # plt.clabel(CS_true, fmt="%1.2f")

    labels = ["Inferred Posterior", "True Posterior"]
    lines = [CS_inferred.collections[0], CS_true.collections[0]]
    plt.legend(lines, labels)
    plt.xlabel("Mu")
    plt.ylabel("Tao")

    plt.title(title)
    plt.savefig(f'{filename}.png')


def main():
    np.random.seed(1)
    # Init parameters
    a0, b0, mu0, lambda0 = 1, 1, 1, 1
    Etao = 1
    threshold = 1e-300
    # Generate data
    mu, tao, N = 0, 1, 1000
    mu_axis = np.linspace(-0.05, 0.15, 100)
    tao_axis = np.linspace(0.9, 1.2, 100)
    X, true_posterior = generate_data(
        mu, tao, N, mu_axis, tao_axis, lambda0, mu0, a0, b0)

    VI_model = VI(a0, b0, mu0, lambda0, Etao, threshold)
    VI_model.fit(X)
    VI_posterior = VI_model.get_posterior(mu_axis, tao_axis)

    # a0-b0-mu0-lambda0-Etao-thres-mu-tao-N
    filename = f'{a0}-{b0}-{mu0}-{lambda0}-{str(threshold)}-{mu}-{tao}-{N}'
    title = f'[a0={a0} b0={b0} mu0={mu0} lambda0={lambda0}] N({mu},{tao}) N={N}'

    plot_contours(mu_axis, tao_axis, VI_posterior,
                  true_posterior, filename, title)


if __name__ == "__main__":
    main()
