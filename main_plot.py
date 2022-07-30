#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)

if __name__ == "__main__": # plot for MNIST convex
    numusers = 10
    num_glob_iters = 800
    dataset = "Mnist"
    local_ep = [20]
    lamda = [15]
    learning_rate = [0.001]
    beta =  [1.0]
    batch_size = [100]
    K = [5]
    personal_learning_rate = [0.001]
    algorithms = ["pFedbayes"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
