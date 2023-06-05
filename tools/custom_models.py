import torch
import torch.nn as nn

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
        y =((W1 @ x) + b1) + ((W2 @ h) + b2)
        if self.actfunc is not None:
            y = self.actfunc(y)
        y = y.squeeze(-1)
        return y, y[:,-1,:].unsqueeze(1)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size}, nonlinearity={repr(self.nonlinearity)})"
    


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        input_size = int(input_size)
        hidden_size = int(hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        def input_weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.input_size, dtype=dtype, device=device))

        def weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, dtype=dtype, device=device))

        def bias():
            return nn.Parameter(torch.zeros(self.hidden_size, dtype=dtype, device=device))

        self.W_ii = input_weight()
        self.W_if = input_weight()
        self.W_ig = input_weight()
        self.W_io = input_weight()

        self.W_hi = weight()
        self.W_hf = weight()
        self.W_hg = weight()
        self.W_ho = weight()

        self.b_ii = bias()
        self.b_if = bias()
        self.b_ig = bias()
        self.b_io = bias()

        self.b_hi = bias()
        self.b_hf = bias()
        self.b_hg = bias()
        self.b_ho = bias()

    def forward(self, x: torch.Tensor, hidden=None) -> tuple:
        sigm = torch.sigmoid
        tanh = torch.tanh

        if hidden is None:
            h_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
            c_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_prev, c_prev = hidden

        i_t = sigm(self.W_ii @ x + self.b_ii + self.W_hi @ h_prev + self.b_hi)
        f_t = sigm(self.W_if @ x + self.b_if + self.W_hf @ h_prev + self.b_hf)
        g_t = tanh(self.W_ig @ x + self.b_ig + self.W_hg @ h_prev + self.b_hg)
        o_t = sigm(self.W_io @ x + self.b_io + self.W_ho @ h_prev + self.b_ho)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)

        return h_t, (h_t, c_t)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size})"