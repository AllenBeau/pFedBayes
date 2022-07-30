import copy
from torch.autograd import Variable
from FLAlgorithms.trainmodel.models import *
from FLAlgorithms.users.userbase import User


class UserpFedBayes(User):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, personal_learning_rate, device, output_dim=10):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device, output_dim=output_dim)

        self.output_dim = output_dim
        self.batch_size = batch_size
        self.N_Batch = len(train_data) // batch_size
        self.personal_learning_rate = personal_learning_rate
        self.optimizer1 = torch.optim.Adam(self.personal_model.parameters(), lr=self.personal_learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        N_Samples = 1
        Round = 5
        self.model.train()
        self.personal_model.train()

        for epoch in range(1, self.local_epochs + 1):

            X, Y = self.get_next_train_batch()
            batch_X = Variable(X.view(self.batch_size, -1))
            batch_Y = Variable(Y.view(self.batch_size, -1))
            label_one_hot = F.one_hot(batch_Y, num_classes=self.output_dim).squeeze(dim=1)

            for r in range(1, Round + 1):
                ### personal model
                epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                layer_params1 = self.personal_model.transform_gaussian_samples(
                    self.personal_model.mus, self.personal_model.rhos, epsilons)

                personal_output = self.personal_model.net(batch_X, layer_params1)
                # calculate the loss
                personal_loss = self.personal_model.combined_loss_personal(
                    personal_output, label_one_hot, layer_params1,
                    self.personal_model.mus, self.personal_model.sigmas,
                    copy.deepcopy(self.model.mus),
                    [t.clone().detach() for t in self.model.sigmas], self.local_epochs)

                self.optimizer1.zero_grad()
                personal_loss.backward()
                self.optimizer1.step()


            ### local model
            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            layer_params2 = self.model.transform_gaussian_samples(self.model.mus, self.model.rhos, epsilons)
            model_output = self.model.net(batch_X, layer_params2)
            # calculate the loss
            model_loss = self.model.combined_loss_local(
                [t.clone().detach() for t in layer_params1],
                copy.deepcopy(self.personal_model.mus),
                [t.clone().detach() for t in self.personal_model.sigmas],
                self.model.mus, self.model.sigmas, self.local_epochs)

            self.optimizer2.zero_grad()
            model_loss.backward()
            self.optimizer2.step()

        return LOSS
