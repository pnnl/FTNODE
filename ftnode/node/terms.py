import numpy as np
import torch 
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(
            self, 
            dims, 
            activation=torch.nn.SiLU(), 
            dtype=torch.float, 
            init_type:str|None=None,
            nonlinearity:str='relu' 
    ):
        # Base constructor
        super().__init__()

        # Store values
        self.dims = list(dims)
        self.activation = activation
        self.dtype = dtype
        self.init_type = init_type
        self.nonlinearity = nonlinearity

        # Create layers
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim).to(self.dtype)
            for input_dim, output_dim in zip(self.dims[:-1], self.dims[1:])
        ])

        if init_type is not None:
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.nonlinearity)
            elif self.init_type == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif self.init_type == "uniform":
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif self.init_type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init_type '{self.init_type}'")

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class FeluSigmoidMLP(nn.Module):
    def __init__(
            self, 
            dims, 
            activation=torch.nn.SiLU(),
            lower_bound=-1,
            upper_bound=0,
            init_type:str|None=None,
            nonlinearity:str='relu'
        ):
        super().__init__()
        self.dims = dims
        self.activation = activation
        self.network = MLP(
            self.dims,
            activation = self.activation,
            init_type=init_type,
            nonlinearity=nonlinearity
        )

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
    def forward(self,x):
        a = self.args["lower_bound"]
        b = self.args["upper_bound"]
        return a + (b-a)*torch.sigmoid(self.network(x))
    



class GeluSigmoidMLP(nn.Module):
    def __init__(
        self,
        dims,
        activation = torch.nn.SiLU(),
        lower_bound=0,
        upper_bound=1,
        init_type:str|None=None,
        nonlinearity:str='relu'
    ):
        super().__init__()

        self.dims = dims
        self.activation = activation
        self.network = MLP(
            self.dims,
            activation = self.activation,
            init_type=init_type,
            nonlinearity=nonlinearity
        )

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    def forward(self,x,u):
        xu = torch.cat([x,u],dim=-1)
        a = self.args['lower_bound']
        b = self.args['upper_bound']

        return a + (b-a)*torch.sigmoid(self.network(xu))
    


class GeluSigmoidMLPfeaturized(nn.Module):
    def __init__(
        self,
        dims,
        activation = torch.nn.SiLU(),
        lower_bound=0,
        upper_bound=1,
        feat_lower_bound = 0,
        feat_upper_bound = 1,
        freq_sample_step = 5,
        init_type:str|None=None,
        nonlinearity:str='relu'
    ):
        super().__init__()

        self.dims = dims
        self.activation = activation

        self.network = MLP(
            self.dims,
            activation = self.activation,
            init_type=init_type,
            nonlinearity=nonlinearity
        )

        self.freq_sample_step = freq_sample_step
        self.featurization_dim = dims[0] - 1
        self.freqs = torch.arange(
            self.freq_sample_step,
            self.featurization_dim*self.freq_sample_step,
            self.freq_sample_step
        )
        

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "feat_lower_bound": feat_lower_bound,
            "feat_upper_bound": feat_upper_bound
        }

    def forward(self,x,u):
        a = self.args['lower_bound']
        b = self.args['upper_bound']
        a_feat = self.args['feat_lower_bound']
        b_feat = self.args['feat_upper_bound']
        x_feats = [x]
        for fq in self.freqs:
            x_feats.append(torch.cos(fq**2*3.14*(x- a_feat)/(b_feat-a_feat)))
        xf = torch.cat(x_feats,dim=-1)
        xu = torch.cat([xf,u],dim=-1)


        return a + (b-a)*torch.sigmoid(self.network(xu))
    
class FeluSigmoidMLPfeaturized(nn.Module):
    def __init__(
        self,
        dims,
        activation = torch.nn.SiLU(),
        lower_bound=-1,
        upper_bound=0,
        feat_lower_bound = 0,
        feat_upper_bound = 1,
        freq_sample_step = 5,
        init_type:str|None=None,
        nonlinearity:str='relu'
    ):
        super().__init__()

        self.dims = dims
        self.activation = activation
        self.network = MLP(
            self.dims,
            activation = self.activation,
            init_type=init_type,
            nonlinearity=nonlinearity
        )

        self.freq_sample_step = freq_sample_step
        self.featurization_dim = dims[0] 
        self.freqs = torch.arange(
            self.freq_sample_step,
            self.featurization_dim*self.freq_sample_step,
            self.freq_sample_step
        )
        

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "feat_lower_bound": feat_lower_bound,
            "feat_upper_bound": feat_upper_bound
        }

    def forward(self,x):
        a = self.args['lower_bound']
        b = self.args['upper_bound']
        a_feat = self.args["feat_lower_bound"]
        b_feat = self.args["feat_upper_bound"]
        x_feats = [x]
        for fq in self.freqs:
            x_feats.append(torch.cos(fq**2*3.14*(x- a_feat)/(b_feat-a_feat)))
        xf = torch.cat(x_feats,dim=-1)

        return a + (b-a)*torch.sigmoid(self.network(xf))



class FTNODE(nn.Module):
    def __init__(self,f,g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, t, x, u_func):
        """
        t: scalar or shape [B]
        x: shape [B, d]
        """
        u = u_func(t)  # shape [B, d_u]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, d]
        if u.dim() == 1:
            u = u.unsqueeze(0)  # [1, d_u]
        fx = self.f(x)          # shape [B, d]
        gx = self.g(x, u)       # shape [B, d]
        return fx * (x - gx)    # shape [B, d]
    