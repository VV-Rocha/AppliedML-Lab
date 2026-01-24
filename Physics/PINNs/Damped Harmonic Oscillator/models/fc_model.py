import torch
from torch import nn

class FCNet(nn.Module):
    def __init__(
        self,
        Nin,
        Nout,
        nlayers,
        nnodes,
        activation,
    ):
        super().__init__()
        self.Nin = Nin
        self.Nout = Nout
        self.nlayers = nlayers
        self.nnodes = nnodes
        self.activation = activation

        self.gen_model()
        
    def gen_model(self,):
        self.layers = [nn.Linear(self.Nin, self.nnodes),
                       self.activation(),
                       ]
        
        for _ in range(self.nlayers-1):
            self.layers.append(nn.Linear(self.nnodes, self.nnodes))
            self.layers.append(self.activation())
        
        self.layers.append(nn.Linear(self.nnodes, self.Nout))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.net(x)
        return x
    
class PINNs_FCNet(FCNet):
    def __init__(
        self,
        gamma0=1.,
        kappa0=1.,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.gamma = nn.Parameter(torch.tensor(gamma0))
        self.kappa = nn.Parameter(torch.tensor(kappa0))