import numpy as np
import math


class CategoricalCategoricalEM:
    def __init__(self, pi_1_shape, pi_2_shape, threshold):
        # pi_1: first categorical dist, pi_2: second cat dist
        self.pi_1_shape = pi_1_shape
        self.pi_2_shape = pi_2_shape
        self.threshold = threshold

    def fit(self, observations):
        self.init_params()
        old_log_likelihood = 0
        iter = 0

        print('\tFitting Expectation Maximization...')
        while True:
            self.E_step(observations)
            self.M_step()
            new_log_likelihood = self.log_likelihood()
            if math.isnan(new_log_likelihood):
                break
            if abs(old_log_likelihood - new_log_likelihood) < self.threshold:
                break
            old_log_likelihood = new_log_likelihood
            iter = iter + 1

            print(new_log_likelihood)

        self.log_likelihood = new_log_likelihood
        print(f'\tIterations= {iter}')

    def init_params(self):
        # initial guesses
        self.pi_1 = np.ones(self.pi_1_shape)*(1/6)  # dice from table
        self.pi_2 = np.ones(self.pi_2_shape)*(1/6)  # dice from players

        self.pi_1[:, 0:4] = 0.09
        self.pi_1[:, 5] = 0.55

        self.pi_2[:, 0] = 0.55
        self.pi_2[:, 1:] = 0.09

        print(
            f"Initial guesses:\n\tTable Dist:\n\t{self.pi_1}\n\tPlayer Dist:{self.pi_2}")

    def get_params(self):
        return self.pi_1, self.pi_2

    def E_step(self, observations):
        # Calculate responsibilities
        K, N = self.pi_1.shape[0], self.pi_2.shape[0]
        n_cat = (self.pi_1.shape[1], self.pi_2.shape[1])

        self.r_nk = np.zeros((K, N, n_cat[0], n_cat[1]))

        den = np.zeros((K, N))
        num = np.zeros((K, N, n_cat[0], n_cat[1]))

        for k in range(K):
            for n in range(N):
                for t in range(n_cat[0]):
                    for m in range(n_cat[1]):
                        s = observations[n][k]
                        if (m+1 + t+1 == s):
                            num[k][n][t][m] = self.pi_1[k][m]*self.pi_2[n][t]
                            den[k][n] += self.pi_1[k][m]*self.pi_2[n][t]
        for k in range(K):
            for n in range(N):
                for t in range(6):
                    for m in range(6):
                        s = observations[n][k]
                        if (m+1 + t+1 == s):
                            self.r_nk[k][n][t][m] = num[k][n][t][m] / den[k][n]

    def M_step(self):
        # Update categorical distributions: pi_1 and pi_2
        K, N = self.pi_1.shape[0], self.pi_2.shape[0]
        n_cat = (self.pi_1.shape[1], self.pi_2.shape[1])
        num_1, num_2 = np.zeros((K, n_cat[0])), np.zeros((N, n_cat[1]))
        den_1, den_2 = np.zeros((K)), np.zeros((N))

        # Update pi_1: table_dist, pi_km
        for k in range(K):
            for m in range(n_cat[0]):
                for n in range(N):
                    for t in range(n_cat[1]):
                        num_1[k][m] += self.r_nk[k][n][t][m]

        for k in range(K):
            for m in range(n_cat[0]):
                den_1[k] += num_1[k][m]

        for k in range(K):
            for m in range(n_cat[0]):
                self.pi_1[k][m] = num_1[k][m]/den_1[k]

        # Update pi_2: player_dist, phi_nt
        for n in range(N):
            for t in range(n_cat[1]):
                for k in range(K):
                    for m in range(n_cat[0]):
                        num_2[n][t] += self.r_nk[k][n][t][m]

        for n in range(N):
            for t in range(n_cat[1]):
                den_2[n] += num_2[n][t]

        for n in range(N):
            for t in range(n_cat[1]):
                self.pi_2[n][t] = num_2[n][t]/den_2[n]

    def log_likelihood(self):
        log_likelihood = 0
        K, N = self.pi_1.shape[0], self.pi_2.shape[0]
        n_cat = (self.pi_1.shape[1], self.pi_2.shape[1])
        for n in range(N):
            for k in range(K):
                for t in range(n_cat[1]):
                    for m in range(n_cat[0]):
                        log_likelihood += self.r_nk[k][n][t][m] * \
                            (np.log(self.pi_1[k][m])+np.log(self.pi_2[n][t]))
        return log_likelihood

    def print_params(self):
        print("Params:")
        print("\tTables distribution:")
        for row in self.pi_1:  # each table
            s = "\t\t"
            for p in row:  # prob of each outcome in that table
                s = s + f"|{p:.4f}"
            print(s)
        print("\tPlayers distribution:")
        for row in self.pi_2:  # each player
            s = "\t\t"
            for p in row:  # prob of each outcome in for that player
                s = s + f"|{p:.4f}"
            print(s)
