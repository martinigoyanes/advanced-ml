from casino import Casino
import numpy as np
from em import CategoricalCategoricalEM
import matplotlib.pyplot as plt


def plot_dice_dist(dice, file, title):
    outcomes = ["1", "2", "3", "4", "5", "6"]
    dice = np.mean(dice, axis=0)
    fig, ax = plt.subplots()
    ax.bar(outcomes, dice, label='Real Dice')
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Outcomes")
    ax.legend()
    plt.savefig(f'fig/{file}')


def plot_dice_em_dist(realDice, emDice, file, title):
    width = 0.3
    outcomes_real = [1, 2, 3, 4, 5, 6]
    outcomes_em = [o+width for o in outcomes_real]
    realDice = np.mean(realDice, axis=0)
    emDice = np.mean(emDice, axis=0)

    fig, ax = plt.subplots()

    ax.bar(outcomes_real, realDice, width=width,
           color='b', align='center', label='Real Dice')
    ax.bar(outcomes_em, emDice, width=width,
           color='r', align='center', label='EM Dice')
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Outcomes")
    ax.set_xticks(outcomes_real)

    ax.legend()
    fig.tight_layout()

    plt.savefig(f'fig/{file}')


def main():
    # np.random.seed(1)
    nTables = 5
    nPlayers = 3
    tableProb = {
        # 50% chance of being prime and same of not being prime table_1
        'init': 1/2,
        'change': 3/4
    }

    primedTableDice = np.ones((nTables, 6)) * (1/6)
    notPrimedTableDice = np.ones((nTables, 6)) * (1/6)
    playerDice = np.ones((nPlayers, 6)) * (1/6)  # one dice per player

    primedTableDice[:, 0] = 0.08  # 0.3
    primedTableDice[:, 1] = 0.08  # 0.05
    primedTableDice[:, 2] = 0.08  # 0.15
    primedTableDice[:, 3] = 0.08  # 0.2
    primedTableDice[:, 4] = 0.08  # 0.1
    primedTableDice[:, 5] = 0.6  # 0.2

    notPrimedTableDice = primedTableDice

    playerDice[:, 0] = 0.6  # 0.1
    playerDice[:, 1] = 0.08  # 0.2
    playerDice[:, 2] = 0.08  # 0.3
    playerDice[:, 3] = 0.08  # 0.2
    playerDice[:, 4] = 0.08  # 0.1
    playerDice[:, 5] = 0.08  # 0.1

    print(f"Tables distribution:\n{notPrimedTableDice}")
    print(f"Players distribution:\n{playerDice}")

    plot_dice_dist(notPrimedTableDice, "table-dice.png",
                   "Mean tables' dice distribution")
    plot_dice_dist(playerDice, "player-dice",
                   "Mean players' dice distribution")

    casino = Casino(nTables, nPlayers, tableProb,
                    primedTableDice, notPrimedTableDice, playerDice)
    casino.run()
    observations = casino.S
    print(f"Observations:\n {observations}")

    em = CategoricalCategoricalEM(
        notPrimedTableDice.shape, playerDice.shape, threshold=1e-4)
    em.fit(observations)
    em.print_params()

    emTableDice, emPlayerDice = em.get_params()
    plot_dice_em_dist(notPrimedTableDice, emTableDice, "em-biased-table.png",
                      "Mean real vs EM's tables' dice distribution")
    plot_dice_em_dist(playerDice, emPlayerDice, "em-biased-player.png",
                      "Mean real vs EM's players' dice distribution")


if __name__ == "__main__":
    main()
