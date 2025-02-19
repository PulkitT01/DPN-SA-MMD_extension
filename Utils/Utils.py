"""
MIT License

Copyright (c) 2020 Shantanu Ghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd
import sklearn.model_selection as sklearn
import torch
from torch.distributions import Bernoulli


class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def convert_to_col_vector(np_arr):
        return np_arr.reshape(np_arr.shape[0], 1)

    @staticmethod
    def test_train_split(covariates_X, treatment_Y, split_size=0.8):
        return sklearn.train_test_split(covariates_X, treatment_Y, train_size=split_size)

    @staticmethod
    def convert_to_tensor(X, Y):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_DCN(X, ps_score, Y_f):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_ps_score = ps_score
        tensor_y_f = Y_f
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score,
                                                           tensor_y_f)
        return processed_dataset

    @staticmethod
    def create_tensors_from_tuple_test(group, t):
        np_df_X = group[0]
        np_ps_score = group[1]
        np_df_Y_f = group[2]
        np_df_e = group[3]
        tensor = Utils.convert_to_tensor_DCN_test(np_df_X, np_ps_score,
                                                  np_df_Y_f, t, np_df_e)

        return tensor

    @staticmethod
    def convert_to_tensor_DCN_test(X, ps_score, Y_f, t, e):
        # Convert inputs to tensors if necessary
        tensor_x = torch.stack([torch.Tensor(i) for i in X]) if not isinstance(X, torch.Tensor) else X
        tensor_ps_score = torch.tensor(ps_score, dtype=torch.float32) if not isinstance(ps_score, torch.Tensor) else ps_score
        tensor_y_f = torch.tensor(Y_f, dtype=torch.float32) if not isinstance(Y_f, torch.Tensor) else Y_f
        tensor_t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
        tensor_e = torch.tensor(e, dtype=torch.float32) if not isinstance(e, torch.Tensor) else e
    
        # Ensure sizes match
        min_size = min(
            tensor_x.size(0), tensor_ps_score.size(0), tensor_y_f.size(0), tensor_t.size(0), tensor_e.size(0)
        )
        tensor_x = tensor_x[:min_size]
        tensor_ps_score = tensor_ps_score[:min_size]
        tensor_y_f = tensor_y_f[:min_size]
        tensor_t = tensor_t[:min_size]
        tensor_e = tensor_e[:min_size]
    
        # Create dataset
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score, tensor_y_f, tensor_t, tensor_e)
        return processed_dataset


    @staticmethod
    def concat_np_arr(X, Y, axis=1):
        return np.concatenate((X, Y), axis)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def get_shanon_entropy(prob):
        if prob < 0:
            return
        if prob == 1:
            return -(prob * math.log(prob))
        elif prob == 0:
            return -((1 - prob) * math.log(1 - prob))
        else:
            return -(prob * math.log(prob)) - ((1 - prob) * math.log(1 - prob))

    @staticmethod
    def get_dropout_probability(entropy, gama=1):
        return 1 - (gama * 0.5) - (entropy * 0.5)

    @staticmethod
    def get_dropout_mask(prob, x):
        return Bernoulli(torch.full_like(x, 1 - prob)).sample() / (1 - prob)

    @staticmethod
    def KL_divergence(rho, rho_hat, device):
        # sigmoid because we need the probability distributions
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
        rho = torch.tensor([rho] * len(rho_hat)).to(device)
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    @staticmethod
    def get_runs(params):
        """
        Gets the run parameters using cartesian products of the different parameters.
        :param params: different parameters like batch size, learning rates
        :return: iterable run set
        """
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    @staticmethod
    def write_to_csv(file_name, list_to_write):
        pd.DataFrame.from_dict(
            list_to_write,
            orient='columns'
        ).to_csv(file_name)
