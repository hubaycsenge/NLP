import torch
import torch.nn as nn
#models based on: https://github.com/nnaisense/evotorch/blob/9d31d59cfcace0c99de59bf59bfbc7ceb55ae11f/src/evotorch/neuroevolution/net/layers.py#L160
class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = "atan",
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()

        input_size = int(input_size)
        hidden_size = int(hidden_size)
        nonlinearity = str(nonlinearity)

        self.W1 = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtype, device=device))
        self.W2 = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device))
        self.b1 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.b2 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        if nonlinearity == "tanh":
            self.actfunc = torch.tanh
        elif nonlinearity == "atan":
            self.actfunc = torch.atan
        elif nonlinearity == "none":
            self.actfunc = None
        else:
            self.actfunc = getattr(nnf, nonlinearity)
        #print(self.actfunc)

        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> tuple:
        if h is None:
            h = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        W1 = self.W1
        W2 = self.W2
        b1 = self.b1.unsqueeze(-1)
        b2 = self.b2.unsqueeze(-1)
        x = x.unsqueeze(-1)
        h = h.unsqueeze(-1)
        '''print('params for X')
        print(W1.shape,x.shape,b1.shape)
        print('params for h')
        print(W2.shape,h.shape,b2.shape)'''
        y =((W1 @ x) + b1) + ((W2 @ h) + b2)
        if self.actfunc is not None:
            y = self.actfunc(y)
        y = y.squeeze(-1)
        return y, y[:,-1,:].unsqueeze(1)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size}, nonlinearity={repr(self.nonlinearity)})"