import flax.linen as nn

from network_parameters import P, Q


class FNOBlock(nn.Module):
    width: int
    activation: callable

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)

    def __call__(self, v):
        return self.activation(self.W(v))


class FNO1D(nn.Module):
    activation: callable
    dv: int
    n_blocks: int = 4

    def setup(self):
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.blocks = [
            FNOBlock(width=self.dv, activation=self.activation)
            for _ in range(self.n_blocks)
        ]
        self.projection = nn.Dense(features=P + Q + 1, use_bias=True)

    # w : fenêtres stencil(u0, P, Q), (n, P+Q+1) -> coefficients c_{j,i}, (n, P+Q+1)
    def __call__(self, w):
        x = self.lifting(w)
        for block in self.blocks:
            x = block(x)
        return self.projection(x)


model = FNO1D(activation=nn.gelu, dv=64)
