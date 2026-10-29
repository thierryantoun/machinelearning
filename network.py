import flax.linen as nn
import jax.numpy as jnp

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
    p: int = P   # points à gauche du point central
    q: int = Q   # points à droite

    def setup(self):
        # lifting : conv de noyau p+q+1, 1 canal -> dv canaux
        self.lifting = nn.Conv(features=self.dv, kernel_size=(self.p + self.q + 1,), padding="VALID", use_bias=True)
        self.blocks = [
            FNOBlock(width=self.dv, activation=self.activation)
            for _ in range(self.n_blocks)
        ]
        self.projection = nn.Dense(features=1, use_bias=True)

    # couches ponctuelles après le lifting : (n, dv) -> (n,)
    def _head(self, x):
        for block in self.blocks:
            x = block(x)
        return self.projection(x)[..., 0]

    # u : (n,) -> N, (n,) ; p cellules fantomes à gauche et q à droite pour la périodicité
    def __call__(self, u):
        u = jnp.pad(u, (self.p, self.q), mode="wrap")
        return self._head(self.lifting(u[..., None]))

    # c : (n,) -> N évalué en j sur la fenêtre constante (c_j, ..., c_j), (n,)
    # la conv d'une fenêtre constante vaut c_j * sum_k kernel[k] + bias
    def constant(self, c):
        kernel = self.lifting.variables["params"]["kernel"]   # (p+q+1, 1, dv)
        bias   = self.lifting.variables["params"]["bias"]     # (dv,)
        return self._head(c[:, None] * jnp.sum(kernel, axis=(0, 1)) + bias)


model = FNO1D(activation=nn.gelu, dv=64)
