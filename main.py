from GridWorld import GridWorld
from DynaQPlus import DynaQPlus
from DynaQ import DynaQ, DynaQType
import matplotlib.pylab as plt
import random

OPEN = 0
BLOCKED = 1
START = 2
GOAL = 3

def main():
    # experiment settings
    numRows = 6
    numCols = 6
    gridWorld = GridWorld(numRows, numCols, (0, 0), (0, 0))
    gridWorld.buildMap("in.txt")
    print(gridWorld.map)

    # run experiment
    type = DynaQType.NORMAL
    numEpisodes = 300
    n = 10
    epsilon = 0.05
    stepSize = 0.5
    kappa = 0.01

    # Normal DynaQ (no exploration bonus)
    dynaQ = DynaQ(type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQ.learn()

    # DynaQ+ with updates
    type = DynaQType.DYNA_Q_PLUS
    dynaQP = DynaQ(type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQP.learn()

    # DynaQ+ NO UPDATE
    type = DynaQType.DYNA_Q_PLUS_NO_UPDATE
    dynaQPNoUpdate = DynaQ(type, numEpisodes, n, epsilon, stepSize, kappa, gridWorld)
    dynaQPNoUpdate.learn()

    # plot results
    plotResults(dynaQ, dynaQP, dynaQPNoUpdate)
    print(dynaQP.q)
    print()
    print(dynaQPNoUpdate.q)
    print()
    print(dynaQ.q)


def plotResults(dynaQ, dynaQP, dynaQPNoUpdate):
    fig, ax = plt.subplots(2, 2)
    # plot cumulative rewards between ALL algorithms
    ax[0][0].set(xlabel="Timesteps", ylabel="Cumulative Reward", title="DynaQ vs. DynaQ+ vs. DynaQ+NoUpdate")
    ax[0][0].plot(dynaQ.cumulativeRewards, color='r', label="DynaQ")
    ax[0][0].plot(dynaQP.cumulativeRewards, color='b', label="DynaQ+")
    ax[0][0].plot(dynaQPNoUpdate.cumulativeRewards, color='g', label="DynaQ+NoUpdate")
    ax[0][0].legend(loc="upper left")

    # plot steps per episodes between ALL algorithms
    ax[0][1].set(xlabel="Episodes", ylabel="Steps Per Episode", title="DynaQ vs. DynaQ+ vs. DynaQ+NoUpdate")
    ax[0][1].plot(dynaQ.stepsPerEp, color='r', label="DynaQ")
    ax[0][1].plot(dynaQP.stepsPerEp, color='b', label="DynaQ+")
    ax[0][1].plot(dynaQPNoUpdate.stepsPerEp, color='g', label="DynaQ+NoUpdate")
    ax[0][1].legend(loc="upper left")

    # plot DynaQPlus vs DynaQPlusNoUpdate cumulative rewards
    ax[1][0].set(xlabel="Timesteps", ylabel="Cumulative Reward", title="DynaQ+ vs. DynaQ+NoUpdate")
    ax[1][0].plot(dynaQP.cumulativeRewards, color='b', label="DynaQ+")
    ax[1][0].plot(dynaQPNoUpdate.cumulativeRewards, color='g', label="DynaQ+NoUpdate")
    ax[1][0].legend(loc="upper left")

    # plot DynaQPlus vs DynaQPlusNoUpdate steps per episode
    ax[1][1].set(xlabel="Episodes", ylabel="Steps Per Episode", title="DynaQ+ vs. DynaQ+NoUpdate")
    ax[1][1].plot(dynaQP.stepsPerEp, color='b', label="DynaQ+")
    ax[1][1].plot(dynaQPNoUpdate.stepsPerEp, color='g', label="DynaQ+NoUpdate")
    ax[1][1].legend(loc="upper left")


    # plot DynaQPlus vs DynaQPlusNoUpdate steps per episode
    plt.show()


if __name__ == "__main__":
    main()
