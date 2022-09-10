import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from casino import Casino


def compute_beta(observations, nTypeTables, casino):
    beta_matrix = np.zeros((len(observations), nTypeTables))
    playerDice = casino.playerDice[0]
    for k in range(len(observations)):
        for i in range(0, 6):
            for j in range(0, 6):
                if i+1 + j+1 == observations[k]:
                    beta_matrix[k][0] += playerDice[i] * casino.notPrimeTableDice[k][j]
                    beta_matrix[k][1] += playerDice[i] * casino.primeTableDice[k][j]

    return beta_matrix


def compute_alpha(observations, casino, nTypeTables, A, beta_matrix):
    # table of alphas
    alpha = np.zeros((len(observations), nTypeTables))
    for i in range(nTypeTables):
        alpha[0][i] = casino.tableProb['init'] * beta_matrix[0][i]

    for k in range(1, len(observations)):
        for i in range(nTypeTables):
                alpha[k][i] += alpha[k-1][0] * A[0][i] * beta_matrix[k][i]
                alpha[k][i] += alpha[k-1][1] * A[1][i] * beta_matrix[k][i]
    return alpha


def posteriorSampling(obs, nTypeTables, alpha_matrix, A):
    tables = []
    sum = 0
    for type in range(nTypeTables):
        sum += alpha_matrix[len(obs)-1][type]
    tableProb = alpha_matrix[len(obs)-1][1]/sum
    lastTable = np.random.binomial(1, tableProb)

    tables.append(lastTable)

    for i in reversed(range(len(obs)-1)):
        prevTable = []
        sum = 0
        for beforeTable in range(nTypeTables):
            prevTable.append(A[beforeTable][lastTable]*alpha_matrix[i][beforeTable])
        for type in range(nTypeTables):
            sum += prevTable[type]
        tableProb = prevTable[1]/sum
        tables.append(np.random.binomial(1, tableProb))
        lastTable = tables[-1]

    # put in correct chronological order
    tables = tables[::-1]
    return tables


def main():
    nTables = 15
    nPlayers = 1
    tableProb = {
        # 50% chance of being prime and same of not being prime table_1
        'init': 1/2,
        'change': 3/4
    }
    A = [[1/4, 3/4], [3/4, 1/4]]
    nTypeTables = len(A[0])

    primedTableDice = np.ones((nTables, 6)) * (1/6)
    notPrimedTableDice = np.ones((nTables, 6)) * (1/6)
    playerDice = np.ones((nPlayers, 6)) * (1/6)  # one dice per player

    primedTableDice[:, 0] = 1
    primedTableDice[:, 1:5] = 0

    casino = Casino(nTables, nPlayers, tableProb,
                    primedTableDice, notPrimedTableDice, playerDice)
    casino.run()
    observations = casino.S[0]
    print(f"Observations: {observations}")
    print(f"Real posterior: {casino.tables}")

    # Sample sequence of table from the given observations
    beta_matrix = compute_beta(observations, nTypeTables, casino)
    alpha_matrix = compute_alpha(observations, casino, nTypeTables, A, beta_matrix)
    tableSequence = posteriorSampling(
        observations, nTypeTables, alpha_matrix, A)
    print(f"Sampled posterior: {tableSequence}")


main()
