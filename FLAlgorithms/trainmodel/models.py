import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class pBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=-3, zeta=10):
        super(pBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = 1
        self.mean_prior = 10
        self.sigma_prior = 5
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)

        for shape in self.layer_param_shapes:
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)

    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                W_shape = (self.hidden_dim, self.output_dim)
                b_shape = (self.output_dim,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos):
        return [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        # compute softplus for variance
        self.sigmas = self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)): samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples

    def sample_epsilons(self, param_shapes):
        epsilons = [torch.normal(mean=torch.zeros(shape), std=0.001*torch.ones(shape)).to(self.device) for shape in
                    param_shapes]
        return epsilons

    def net(self, X, layer_params):
        layer_input = X
        for i in range(len(layer_params) // 2 - 1):
            h_linear = torch.mm(layer_input, layer_params[2 * i]) + layer_params[2 * i + 1]
            layer_input = F.relu(h_linear)

        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        return output

    def log_softmax_likelihood(self, yhat_linear, y):
        return torch.nansum(y * F.log_softmax(yhat_linear), dim=0)

    def combined_loss_personal(self, output, label_one_hot, params, mus, sigmas, mus_local, sigmas_local, num_batches):

        # Calculate data likelihood
        log_likelihood_sum = torch.sum(self.log_softmax_likelihood(output, label_one_hot))
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]),
                            Normal(mus_local[i].detach(), sigmas_local[i].detach())))  for i in range(len(params))])

        return 1.0 / num_batches * (self.zeta * KL_q_w) - log_likelihood_sum

    def combined_loss_local(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()),
                        Normal(mus_local[i], sigmas_local[i]))) for i in range(len(params))])
        return 1.0 / num_batches * (self.zeta * KL_q_w)
