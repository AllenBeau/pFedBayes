import torch
from tqdm import tqdm

from FLAlgorithms.users.userpFedbayes import UserpFedBayes
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np


# Implementation for FedAvg Server
class pFedBayes(Server):
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, device, personal_learning_rate,
                 output_dim=10, post_fix_str=''):
        super().__init__(dataset, algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, device)

        # Initialize data for all  users
        data = read_data(dataset)
        self.personal_learning_rate = personal_learning_rate
        self.post_fix_str = post_fix_str
        total_users = len(data[0])
        print('clients initializting...')
        for i in tqdm(range(total_users), total=total_users):
            id, train, test = read_user_data(i, data, dataset, device)
            user = UserpFedBayes(id, train, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer,
                                 personal_learning_rate, device, output_dim=output_dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        acc = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.send_parameters()
            # Evaluate model each interation
            self.evaluate_bayes()

            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs)
            self.aggregate_parameters()

        self.save_results(self.post_fix_str)
        return self.save_model(self.post_fix_str)

    def evaluate_bayes(self):
        stats = self.testpFedbayes()
        stats_train = self.train_error_and_loss_pFedbayes()
        per_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        glob_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_per_acc.append(per_acc)
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)

        print("Average personal Accurancy: ", per_acc)
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)