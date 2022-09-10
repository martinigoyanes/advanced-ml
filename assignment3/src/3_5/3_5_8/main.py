from ..casino import Casino
import matplotlib.pyplot as plt 
import numpy as np

def main():
    nTables = 3
    nPlayers = 1000
    tableProb = {
        # 50% chance of being prime and same of not being prime table_1
        'init': 1/2,
        'change': 3/4
    }

    # Case 1:
    # All dice unbiased

    primedTableDice = np.ones((nTables, 6)) * (1/6)
    notPrimedTableDice = np.ones((nTables, 6)) * (1/6)
    playerDice = np.ones((nPlayers, 6)) * (1/6)  # one dice per player

    casino = Casino(nTables, nPlayers, tableProb,
                    primedTableDice, notPrimedTableDice, playerDice)
    casino.run()

    plt.figure()
    plt.hist(casino.S, stacked=True)
    plt.savefig('3_5_8/unbiased.png')

    # Case 2:
    # Player dice biased towards 1

    primedTableDice = np.ones((nTables, 6)) * (1/6)
    notPrimedTableDice = np.ones((nTables, 6)) * (1/6)
    playerDice = np.ones((nPlayers, 6)) * (1/6)  # one dice per player

    playerDice[:, 0] = 1/2
    playerDice[:, 1:5] = 1/10

    casino = Casino(nTables, nPlayers, tableProb,
                    primedTableDice, notPrimedTableDice, playerDice)
    casino.run()

    plt.figure()
    plt.hist(casino.S, stacked=True)
    plt.savefig('3_5_8/player-biased.png')

    # Case 3:
    # Prime and not prime tables' dice biased towards 1 player's  dice biased towards 6

    primedTableDice = np.ones((nTables, 6)) * (1/6)
    primedTableDice[:, 0] = 1/2
    primedTableDice[:, 1:5] = 1/10

    notPrimedTableDice = np.ones((nTables, 6)) * (1/6)
    notPrimedTableDice[:, 0] = 1/2
    notPrimedTableDice[:, 1:5] = 1/10

    playerDice = np.ones((nPlayers, 6)) * (1/6)  # one dice per player
    playerDice[:, 5] = 1/2
    playerDice[:, 0:4] = 1/10

    casino = Casino(nTables, nPlayers, tableProb,
                    primedTableDice, notPrimedTableDice, playerDice)
    casino.run()

    plt.figure()
    plt.hist(casino.S, stacked=True)
    plt.savefig('3_5_8/all-biased.png')


main()
