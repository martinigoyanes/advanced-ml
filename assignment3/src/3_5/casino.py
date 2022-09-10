import numpy as np
import matplotlib.pyplot as plt



class Casino:
    def __init__(self, nTables, nPlayers, tableProb, primeTableDice, notPrimeTableDice, playerDice):
        self.nTables = nTables
        self.nPlayers = nPlayers
        self.primeTableDice = primeTableDice
        self.notPrimeTableDice = notPrimeTableDice
        self.playerDice = playerDice
        self.tableProb = tableProb
        self.tables = self._init_tables()

    def _init_tables(self):
        tables = []  # Y_k
        firstTable = np.random.binomial(1, self.tableProb['init'], 1)[0]
        tables.append(firstTable)

        for k in range(1, self.nTables):
            if tables[k-1] == 0:  # Y_{k-1} == 0 (not prime)
                t = np.random.binomial(1, self.tableProb['change'], 1)
            else:  # Y_{k-1} == 1 (prime)
                t = np.random.binomial(1, 1-self.tableProb['change'], 1)
            tables.append(t[0])
        return tables

    def _throw_dice(self, dice):
        outcomes = np.random.multinomial(1, dice)
        return np.argmax(outcomes) + 1

    def run(self):
        S_n = []  # S_{k,n} sum of the outcomes from X_{k,n} and Z_{k,n}
        for p in range(self.nPlayers):
            S_k = []
            for k, table in enumerate(self.tables):
                if table == 0:  # Y_k == 0 (not prime table)
                    X_nk = self._throw_dice(self.notPrimeTableDice[k])
                else:  # Y_k == 1 (prime table)
                    X_nk = self._throw_dice(self.primeTableDice[k])

                Z_nk = self._throw_dice(self.playerDice[p])
                S_k.append(X_nk + Z_nk)
            S_n.append(np.asarray(S_k))

        self.S = np.asarray(S_n)

