import torch
import torch.nn as nn
import torch.utils.data


class shallow_net(nn.Module):
    def __init__(self, training_mode, device):
        print("Training mode: {0}".format(training_mode))
        super(shallow_net, self).__init__()
        self.training_mode = training_mode

        # encoder
        self.encoder = nn.Sequential(nn.Linear(in_features=17, out_features=10),
                                     nn.Tanh()
                                     )

        if self.training_mode == "train":
            # decoder
            self.decoder = nn.Sequential(nn.Linear(in_features=10, out_features=17),
                                         nn.Tanh(),
                                         nn.Linear(in_features=17, out_features=17))

    def forward(self, x):
        x = x.float().to(next(self.parameters()).device)
        x = self.encoder(x)
        if self.training_mode == "train":
            x = self.decoder(x)
        return x
