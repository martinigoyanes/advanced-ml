import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, poisson


def load_data(X_file, S_file):
    X, S = [], []

    with open(f"2_4/data/{X_file}", 'r') as f:
        for line in f.readlines():
            X.append(line.split(' '))
        X = np.array(X).astype(float)

    with open(f"2_4/data/{S_file}", 'r') as f:
        for line in f.readlines():
            S.append(line.split()[0])
        S = np.array(S).astype(float)
    return X, S


def plot_contours(X, S, mu_k, covar_k, lambda_k):
    N_points = 300
    N_components = mu_k.shape[0]

    X0_min, X0_max = np.min(X[:, 0]), np.max(X[:, 0])
    X1_min, X1_max = np.min(X[:, 1]), np.max(X[:, 1])

    X0_axis = np.linspace(X0_min, X0_max, N_points)
    X1_axis = np.linspace(X1_min, X1_max, N_points)

    X0, X1 = np.meshgrid(X0_axis, X1_axis)
    plt.figure()
    # plot X points
    plt.scatter(X[:, 0], X[:, 1], s=S)

    # Draw Mixture of Gaussians
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    for k in range(N_components):
        mu = mu_k[k]
        covar = covar_k[k]
        rv = multivariate_normal(mu, covar)
        Z = rv.pdf(np.dstack((X0, X1)))
        plt.contour(X0, X1, Z, colors=colors[k],
                    linewidths=lambda_k[k], alpha=0.4)

    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.title(f"N_components = {N_components}")
    plt.savefig(f"2_4/plots/{N_components}.png")


class GaussianPoissonEM:
    def __init__(self, N_components, threshold):
        self.N_components = N_components
        self.threshold = threshold

    def fit(self, X, S):
        self.init_params(X, S)
        old_log_likelihood = 0
        iter = 0

        print('\tFitting Expectation Maximization...')
        while True:
            self.E_step(X, S)
            self.M_step(X, S)
            new_log_likelihood = self.log_likelihood(X, S)
            if abs(old_log_likelihood - new_log_likelihood) < self.threshold:
                break
            old_log_likelihood = new_log_likelihood
            iter = iter + 1

        self.log_likelihood = new_log_likelihood
        print(f'\tIterations= {iter}')

    def init_params(self, X, S):
        N_points, N_dim = X.shape

        self.pi_k = np.full(self.N_components, 1/self.N_components)
        self.mu_k = X[np.random.choice(
            N_points, self.N_components, replace=False)]
        self.covar_k = np.full(
            (self.N_components, N_dim, N_dim), np.cov(X, rowvar=False))
        self.lambda_k = np.full(self.N_components, np.mean(S))
        self.lambda_k = self.lambda_k / \
            np.array([i for i in range(1, self.N_components+1)])

        # Exercise assumes covariances diagonal so...
        covar_k = []
        for c in self.covar_k:
            covar_k = np.append(covar_k, np.diag(np.diag(c)))
        self.covar_k = np.array(covar_k).reshape(self.N_components, 2, 2)

    def get_params(self):
        return self.pi_k, self.mu_k, self.covar_k, self.lambda_k

    def E_step(self, X, S):
        N_points = X.shape[0]
        r_nk = np.zeros((N_points, self.N_components))

        for n in range(N_points):
            for k in range(self.N_components):
                r_nk[n][k] = self.pi_k[k] *\
                    multivariate_normal(self.mu_k[k], self.covar_k[k]).pdf(X[n]) *\
                    poisson(self.lambda_k[k]).pmf(S[n])
            r_nk[n] = r_nk[n]/r_nk[n].sum()

        # Set responsibility, i.e, E[I(Z_n=class_k)]
        self.r_nk = r_nk

    def M_step(self, X, S):
        N_k = self.r_nk.sum(axis=0)
        N_points, N_dim = X.shape

        for k in range(self.N_components):
            self.pi_k[k] = N_k[k]/N_points
            self.lambda_k[k] = np.sum(self.r_nk[:, k] * S[:])/N_k[k]
            for d in range(N_dim):
                self.mu_k[k][d] = np.sum(self.r_nk[:, k] * X[:, d])/N_k[k]
                self.covar_k[k][d][d] = np.sum(self.r_nk[:, k] *
                                               (X[:, d] - self.mu_k[k][d]) *
                                               np.transpose(X[:, d] - self.mu_k[k][d]))/N_k[k]

    def log_likelihood(self, X, S):
        log_likelihood = np.zeros(self.N_components)
        for k in range(self.N_components):
            log_likelihood[k] = sum(self.r_nk[:, k] * (self.pi_k[k] * multivariate_normal(
                self.mu_k[k], self.covar_k[k]).pdf(X[::]) * poisson(self.lambda_k[k]).pmf(S[:])))
        log_likelihood = np.sum(np.log(log_likelihood))

        return log_likelihood

    def print_params(self):
        for k in range(self.N_components):
            print(f'\tComponent {k}:')
            pi = self.pi_k[k]
            mu_0, mu_1 = self.mu_k[k][0], self.mu_k[k][1]
            c_0, c_1 = self.covar_k[k][0][0], self.covar_k[k][1][1]
            rate = self.lambda_k[k]
            print(
                f'\t\tpi=[{pi:.4f}]\n\t\tNormal([{mu_0:.4f}, {mu_1:.4f}],[{c_0:.4f}, {c_1:.4f}])\n\t\tPoisson({rate})')


def main():
    np.random.seed(1)
    X_file, S_file = "X.txt", "S.txt"
    X, S = load_data(X_file, S_file)
    N_components = [2, 3, 5]
    threshold = 1e-10
    for k in N_components:
        print('#'*10, f'\t{k} Components\t', '#'*10)
        EM_model = GaussianPoissonEM(k, threshold)
        EM_model.fit(X, S)
        EM_model.print_params()

        pi_k, mu_k, covar_k, lambda_k = EM_model.get_params()
        plot_contours(X, S, mu_k, covar_k, lambda_k)


if __name__ == "__main__":
    main()
