from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DCN.DCN import DCN
import os



class DCN_network:

    @staticmethod
    def get_num_workers():
        """
        Dynamically determine the number of workers for DataLoader based on available CPU cores.
        """
        return max(1, os.cpu_count() // 2)  # Use half the available cores, at least 1

    def train(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"]
        treated_set_train = train_parameters["treated_set_train"]
        control_set_train = train_parameters["control_set_train"]

        input_nodes = train_parameters["input_nodes"]

        phases = ['train', 'val']

        print("Saved model path: {0}".format(model_save_path))

        treated_data_loader_train = torch.utils.data.DataLoader(treated_set_train,
                                                                batch_size=treated_batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=0)

        control_data_loader_train = torch.utils.data.DataLoader(control_set_train,
                                                                batch_size=control_batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=0)

        running_mode = train_parameters.get("running_mode", "jobs").lower()
        output_dim = 1 if running_mode == "ihdp" else 2
        network = DCN(training_flag=True, input_nodes=input_nodes, output_dim=output_dim).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        lossF = nn.MSELoss()
        min_loss = 100000.0
        dataset_loss = 0.0
        print(".. Training started ..")
        print(device)

        for epoch in range(epochs):
            network.train()
            total_loss = 0
            train_set_size = 0

            if epoch % 2 == 0:
                dataset_loss = 0
                # train treated
                network.hidden1_Y1.weight.requires_grad = True
                network.hidden1_Y1.bias.requires_grad = True
                network.hidden2_Y1.weight.requires_grad = True
                network.hidden2_Y1.bias.requires_grad = True
                network.out_Y1.weight.requires_grad = True
                network.out_Y1.bias.requires_grad = True

                network.hidden1_Y0.weight.requires_grad = False
                network.hidden1_Y0.bias.requires_grad = False
                network.hidden2_Y0.weight.requires_grad = False
                network.hidden2_Y0.bias.requires_grad = False
                network.out_Y0.weight.requires_grad = False
                network.out_Y0.bias.requires_grad = False

                for batch in treated_data_loader_train:
                    covariates_X, ps_score, y_f = batch

                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    train_set_size += covariates_X.size(0)
                    y1_hat = network(covariates_X, ps_score)[0]

                    # Switch between regression and classification
                    if running_mode == "ihdp":
                        y_f = y_f.float().to(device).view(-1)
                        y1_hat = y1_hat.view(-1)
                        loss = F.mse_loss(y1_hat, y_f)
                    else:
                        y_f = y_f.long().to(device)
                        loss = F.cross_entropy(y1_hat, y_f)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                dataset_loss = total_loss

            elif epoch % 2 == 1:
                # train controlled
                network.hidden1_Y1.weight.requires_grad = False
                network.hidden1_Y1.bias.requires_grad = False
                network.hidden2_Y1.weight.requires_grad = False
                network.hidden2_Y1.bias.requires_grad = False
                network.out_Y1.weight.requires_grad = False
                network.out_Y1.bias.requires_grad = False

                network.hidden1_Y0.weight.requires_grad = True
                network.hidden1_Y0.bias.requires_grad = True
                network.hidden2_Y0.weight.requires_grad = True
                network.hidden2_Y0.bias.requires_grad = True
                network.out_Y0.weight.requires_grad = True
                network.out_Y0.bias.requires_grad = True

                for batch in control_data_loader_train:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    train_set_size += covariates_X.size(0)
                    y1_hat = network(covariates_X, ps_score)[0]

                    # Switch between regression and classification
                    if running_mode == "ihdp":
                        y_f = y_f.float().to(device).view(-1)
                        y1_hat = y1_hat.view(-1)
                        loss = F.mse_loss(y1_hat, y_f)
                    else:
                        y_f = y_f.long().to(device)
                        loss = F.cross_entropy(y1_hat, y_f)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                dataset_loss = dataset_loss + total_loss

            # print("epoch: {0}, train_set_size: {1} loss: {2}".
            #       format(epoch, train_set_size, total_loss))
            if epoch % 10 == 9:
                print("epoch: {0}, Treated + Control loss: {1}".format(epoch, dataset_loss))
            # if epoch % 2 == 1:
            #     print("epoch: {0}, Treated + Control loss: {1}".format(epoch, dataset_loss))
            # if dataset_loss < min_loss:
            #     print("Current loss: {0}, over previous: {1}, Saving model".
            #           format(dataset_loss, min_loss))
            #     min_loss = dataset_loss
        torch.save(network.state_dict(), model_save_path)

    def eval(self, eval_parameters, device, input_nodes):
        print(".. Evaluation started ..")
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        model_path = eval_parameters["model_save_path"]
        running_mode = eval_parameters.get("running_mode", "jobs").lower()

        output_dim = 1 if running_mode == "ihdp" else 2
        network = DCN(training_flag=False, input_nodes=input_nodes, output_dim=output_dim).to(device)

        network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        network.eval()

        treated_data_loader = torch.utils.data.DataLoader(treated_set, shuffle=False, num_workers=0)
        control_data_loader = torch.utils.data.DataLoader(control_set, shuffle=False, num_workers=0)

        ITE_dict_list = []
        y_f_list = []
        y1_hat_list = []
        y0_hat_list = []
        e_list = []
        T_list = []
        predicted_ITE_list = []

        def process_batch(data_loader, is_treated):
            for batch in data_loader:
                covariates_X, ps_score, y_f, t, e = batch
                covariates_X = covariates_X.to(device)
                ps_score = ps_score.squeeze().to(device)
                y_f = y_f.to(device)

                y1_pred, y0_pred = network(covariates_X, ps_score)

                if running_mode == "ihdp":
                    # Regression output: use directly
                    y1_hat = y1_pred.squeeze()
                    y0_hat = y0_pred.squeeze()
                else:
                    # Classification output: take argmax
                    _, y1_hat = torch.max(y1_pred.data, 1)
                    _, y0_hat = torch.max(y0_pred.data, 1)

                predicted_ITE = y1_hat - y0_hat

                ITE_dict_list.append(self.create_ITE_Dict(
                    covariates_X,
                    ps_score.item(),
                    y_f.item(),
                    y1_hat.item(),
                    y0_hat.item(),
                    predicted_ITE.item()
                ))

                y_f_list.append(y_f.item())
                y1_hat_list.append(y1_hat.item())
                y0_hat_list.append(y0_hat.item())
                predicted_ITE_list.append(predicted_ITE.item())
                e_list.append(e.item())
                T_list.append(t)

        process_batch(treated_data_loader, is_treated=True)
        process_batch(control_data_loader, is_treated=False)

        return {
            "predicted_ITE": predicted_ITE_list,
            "ITE_dict_list": ITE_dict_list,
            "y1_hat_list": y1_hat_list,
            "y0_hat_list": y0_hat_list,
            "e_list": e_list,
            "yf_list": y_f_list,
            "T_list": T_list
        }

    @staticmethod
    def create_ITE_Dict(covariates_X, ps_score, y_f,
                        y1_hat,
                        y0_hat,
                        predicted_ITE):
        result_dict = OrderedDict()
        covariate_list = [element.item() for element in covariates_X.flatten()]
        idx = 0
        for item in covariate_list:
            idx += 1
            result_dict["X" + str(idx)] = item

        result_dict["ps_score"] = ps_score
        result_dict["factual"] = y_f
        result_dict["y1_hat"] = y1_hat
        result_dict["y0_hat"] = y0_hat
        result_dict["predicted_ITE"] = predicted_ITE

        return result_dict
