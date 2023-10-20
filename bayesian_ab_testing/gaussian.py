# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import matplotlib.pyplot as plt
import numpy as np


# np.random.seed(2)
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        self.p = p #true mean
        self.mean = 0.1 # estimated mean TODO
        self.precision = 1 # TODO
        self.N = 0 # for information only

    def pull(self):
        return np.random.normal(self.mean, self.precision)

#     def sample(self):
#         return np.random.normal(a,b) # TODO - draw a sample from Beta(a, b)

    def update(self, x):
        self.mean = (self.mean+x)/self.N # TODO ((self.N - 1)*self.p_estimate + x) / self.N
        self.N += 1
        self.mean = ((self.N - 1)*self.self.mean+x) / self.N


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label="real p: {0:.4f}, win rate = {1}/{2}".format(b.p,b.a - 1,b.N))
    plt.title("Bandit distributions after {} trials".format(trial))
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        # Thompson sampling
        j = np.argmax([b.mean for b in bandits]) # TODO

        # plot the posteriors
        if i in sample_points:
            plot(bandits, i)

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

      # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])


if __name__ == "__main__":
    experiment()