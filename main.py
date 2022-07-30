#!/usr/bin/env python
from FLAlgorithms.servers.serverpFedbayes import pFedBayes
from FLAlgorithms.trainmodel.models import *
from utils import argparse
from utils.plot_utils import *
import torch

torch.manual_seed(1)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, device,
         weight_scale, rho_offset, zeta):
    post_fix_str = 'plr_{}_lr_{}'.format(personal_learning_rate, learning_rate)
    model_path = []
    for i in range(times):
        print("---------------Running time:------------", i)
        if model == "pbnn":
            if dataset == "Mnist":
                model = pBNN(784, 100, 10, device, weight_scale, rho_offset, zeta).to(device), model
            else:
                model = pBNN(3072, 100, 10, device, weight_scale, rho_offset, zeta).to(device), model

        server = pFedBayes(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                           local_epochs, optimizer, numusers, i, device, personal_learning_rate,
                           post_fix_str=post_fix_str)

        model_path.append(server.train())
        _, nums_list, acc_list, _ = server.testpFedbayes()

    result_path = average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta=beta, algorithms=algorithm, batch_size=batch_size,
                               dataset=dataset, k=K, personal_learning_rate=personal_learning_rate, times=times,
                               post_fix_str=post_fix_str)
    return model_path, result_path


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist"])
    parser.add_argument("--model", type=str, default="pbnn", choices=["pbnn"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument("--weight_scale", type=float, default=0.1)
    parser.add_argument("--rho_offset", type=int, default=-3)
    parser.add_argument("--zeta", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedBayes",
                        choices=["pFedMe", "FedAvg", "FedBayes"])
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.001,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    return main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        device=device,
        weight_scale=args.weight_scale,
        rho_offset=args.rho_offset,
        zeta=args.zeta
    )


if __name__ == "__main__":
    run()
