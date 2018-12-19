from model import TabularDynaQ, DeepDynaQ
from baseline import QNetwork, run_episodes, test_greedy, train
import random
import torch
import sys
import gym
from helpers import ReplayMemory, smooth
import matplotlib.pyplot as plt
import numpy as np

# parameter settings
env = gym.envs.make("CartPole-v0")
n = 3
learning_rate = 0.5
discount_factor = .8
capacity = 10000
experience_replay = True
true_gradient = False
batch_size = 64
model_batch = False


def run_deep(num_steps, planning_steps):
    memory = ReplayMemory(capacity)
    dynaQ = DeepDynaQ(env,
                      planning_steps=planning_steps, discount_factor=discount_factor, lr=1e-3, epsilon=None,
                      memory=memory,
                      experience_replay=experience_replay, true_gradient=true_gradient, batch_size=batch_size,
                      model_batch=model_batch)
    dynaQ.learn_policy(num_steps)
    train_episode_lengths = dynaQ.episode_lengths.copy()
    dynaQ.test_model_greedy(100)
    test_episode_lengths = dynaQ.episode_lengths.copy()
    return train_episode_lengths, test_episode_lengths


def run_tabular(num_steps, planning_steps):
    dynaQ = TabularDynaQ(env,
                         planning_steps=planning_steps, discount_factor=discount_factor, lr=learning_rate, epsilon=0.2,
                         deterministic=False)
    dynaQ.learn_policy(num_steps)
    train_episode_lengths = dynaQ.episode_lengths.copy()
    dynaQ.test_model_greedy(100)
    test_episode_lengths = dynaQ.episode_lengths.copy()
    return train_episode_lengths, test_episode_lengths


def run_baseline(num_steps):
    batch_size = 64
    discount_factor = 0.8
    learn_rate = 1e-3
    memory = ReplayMemory(10000)
    num_hidden = 128

    model = QNetwork(num_hidden)
    train_lengths, steps = run_episodes(train, model, memory, env, num_steps, batch_size, discount_factor, learn_rate, num_steps)
    test_lengths, mean_duration = test_greedy(model, env, 100, discount_factor, learn_rate)
    return train_lengths, test_lengths


if __name__ == "__main__":

    # number of runs for variance
    runs = 10

    # parameters
    updates = False
    base = False
    # training_steps = [500, 1000, 5000, 10000]
    # planning_steps = [5, 5, 5, 5]
    # training_steps = [5000, 5000, 5000, 5000]
    # planning_steps = [1, 3, 5, 10]
    training_steps = [5000]
    planning_steps = [3]

    # initialize means
    tab_means, deep_means, base_means = [], [], []
    tab_vars, deep_vars, base_vars = [], [], []

    # run stuff
    for i, (steps, n) in enumerate(zip(training_steps, planning_steps)):

        tab_means_inter, deep_means_inter, base_means_inter = [], [], []

        print("Run {} of {}".format(i, len(training_steps) - 1))
        for j in range(runs):

            # We will seed the algorithm (before initializing QNetwork!) for reproducability
            random.seed(i + j)
            torch.manual_seed(i + j)
            env.seed(i + j)

            if not updates:
                # deep_train_lengths, deep_test_lengths = run_deep(steps, n)
                tab_train_lengths, tab_test_lengths = run_tabular(steps, n)
                if base:
                    base_train, base_test_lengths = run_baseline(steps)

                tab_means_inter.append(np.mean(np.array(tab_test_lengths)))
                # deep_means_inter.append(np.mean(np.array(deep_test_lengths)))
            else:
                if base:
                    base_train, base_test_lengths = run_baseline(steps * n + steps)
            if base:
                base_means_inter.append(np.mean(np.array(base_test_lengths)))

        if not updates:
            tab_means.append(np.mean(np.array(tab_means_inter)))
            tab_vars.append(np.std(np.array(tab_means_inter)))
            # deep_means.append(np.mean(np.array(deep_means_inter)))
            # deep_vars.append(np.std(np.array(deep_means_inter)))

        if base:
            base_means.append(np.mean(np.array(base_means_inter)))
            base_vars.append(np.std(np.array(base_means)))

    print("Tabular")
    print("Means")
    print(tab_means)
    print("Vars")
    print(tab_vars)

    print("Deep")
    print("Means")
    print(deep_means)
    print("Vars")
    print(deep_vars)

    print("Base")
    print("Means")
    print(base_means)
    print("Vars")
    print(base_vars)
