import matplotlib.pyplot as plt
import h5py
import numpy as np

plt.rcParams.update({'font.size': 14})


def simple_read_data(alg, parent_path='./results'):
    """
    h5 file read.
    @param parent_path:
    @param alg:
    @return:
    """
    print(alg)
    hf = h5py.File('{}/'.format(parent_path) + '{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_per_acc = np.array(hf.get('rs_per_acc')[:]) if hf.get('rs_per_acc') is not None else np.zeros(
        shape=rs_glob_acc.shape)
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    if len(rs_per_acc) == 0:
        rs_per_acc = [np.nan] * len(rs_glob_acc)
    return rs_train_acc, rs_train_loss, rs_glob_acc, rs_per_acc

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = []):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    per_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" +str(lamb[i])
        if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i]) + "_plr_"+ str(personal_learning_rate[i])  + "_lr_"+ str(learning_rate[i])

        train_acc[i, :], train_loss[i, :], glob_acc[i, :], per_acc[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg"))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    return glob_acc, per_acc, train_acc, train_loss


def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0, beta=0,
                                algorithms="", batch_size=0, dataset="", k=0, personal_learning_rate=0, times=5,
                                post_fix_str=''):
    train_acc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_acc = np.zeros((times, Numb_Glob_Iters))
    rs_per_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list = [algorithms] * times
    for i in range(times):
        string_learning_rate = str(learning_rate)
        string_learning_rate = string_learning_rate + "_" + str(beta) + "_" + str(lamb)
        algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(
            batch_size) + "b"  "_" + str(loc_ep1) + "_" + str(i) + "_" + post_fix_str

        train_acc[i, :], train_loss[i, :], glob_acc[i, :], rs_per_acc[i, :] = np.array(
            simple_read_data(dataset + "_" + algorithms_list[i]))[:, :Numb_Glob_Iters]
    return glob_acc, rs_per_acc, train_acc, train_loss


def average_data(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", beta="", algorithms="",
                 batch_size=0, dataset="", k="", personal_learning_rate="", times=5, post_fix_str=''):
    glob_acc, rs_per_acc, train_acc, train_loss = get_all_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                              learning_rate, beta, algorithms,
                                                                              batch_size, dataset, k,
                                                                              personal_learning_rate, times,
                                                                              post_fix_str)
    glob_acc_data = np.average(glob_acc, axis=0)
    rs_per_acc_data = np.average(rs_per_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    # store average value to h5 file
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())

    print("std:", np.std(max_accurancy))
    print("Mean:", np.mean(max_accurancy))

    alg = dataset + "_" + algorithms
    alg = alg + "_" + str(learning_rate) + "_" + str(beta) + "_" + str(lamb) + "_" + str(num_users) + "u" + "_" + str(
        batch_size) + "b" + "_" + str(loc_ep1)
    if algorithms == "pFedMe" or algorithms == "pFedMe_p":
        alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
    alg = alg + "_" + post_fix_str + "_" + "avg"
    if len(glob_acc) != 0 & len(train_acc) & len(train_loss):
        with h5py.File("./results/" + '{}.h5'.format(alg, loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_per_acc', data=rs_per_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            return hf.filename


def get_label_name(name):
    if name.startswith("pFedMe"):
        if name.startswith("pFedMe_p"):
            return "pFedMe"+ " (PM)"
        else:
            return "pFedMe"+ " (GM)"
    if name.startswith("pFedbayes"):
        return "pFedbayes"
    if name.startswith("PerAvg"):
        return "Per-FedAvg"
    if name.startswith("FedAvg"):
        return "FedAvg"
    if name.startswith("APFL"):
        return "APFL"

def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len<3:
        return data
    for i in range(len(data)):
        x = data[i]
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)


def plot_summary_one_figure_mnist_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta,
                                          algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, per_acc_, train_acc, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters,
                                                               lamb,learning_rate, beta, algorithms_list, batch_size,
                                                               dataset, k, personal_learning_rate)
    for i in range(Numb_Algs):
        print("max accurancy:", glob_acc_[i].max())
    glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    per_acc = average_smooth(per_acc_, window='flat')

    linestyles = ['-', '--', '-.', '-', '--', '-.']
    linestyles = ['-', '-', '-', '-', '-', '-', '-']
    # linestyles = ['-','-','-','-','-','-','-']
    markers = ["o", "v", "s", "*", "x", "P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1, figsize=(5, 5))
    plt.title("$\mu-$" + "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label+'(GM)', linewidth=1, color=colors[i], marker=markers[i],
                 markevery=0.2, markersize=5)

        plt.plot(per_acc[i, 1:], linestyle=linestyles[i+2], label=label+'(PM)', linewidth=1, color=colors[i+2], marker=markers[i+2],
                 markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.10, 0.95])  # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    # plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    plt.close()
