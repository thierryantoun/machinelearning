import flax.linen as nn
import jax.numpy as jnp

from network_parameters import P, Q


class stencil_conv1D_block(nn.Module):
    width: int
    activation: callable

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)

    def __call__(self, v):
        return self.activation(self.W(v))


class stencil_conv1D(nn.Module):
    activation: callable
    dv: int
    n_blocks: int = 4
    p: int = P   # points à gauche du point central
    q: int = Q   # points à droite

    def setup(self):
        # lifting : conv de noyau p+q+1, 3 canaux (rho, rho*u, E) -> dv canaux
        self.lifting = nn.Conv(features=self.dv, kernel_size=(self.p + self.q + 1,), padding="VALID", use_bias=True)
        self.blocks = [
            stencil_conv1D_block(width=self.dv, activation=self.activation)
            for _ in range(self.n_blocks)
        ]
        self.projection = nn.Dense(features=3, use_bias=True)

    # couches ponctuelles après le lifting : (n, dv) -> (3, n)
    def _head(self, x):
        for block in self.blocks:
            x = block(x)
        return self.projection(x).T

    # U : (3, n) -> N, (3, n) ; p cellules fantomes à gauche et q à droite pour la périodicité
    def __call__(self, U):
        U = jnp.pad(U, ((0, 0), (self.p, self.q)), mode="wrap")
        return self._head(self.lifting(U.T))   # U.T : (n+p+q, 3), canaux en dernier pour nn.Conv

    # c : (3, n) -> N évalué en j sur la fenêtre constante (c_j, ..., c_j), (3, n)
    # la conv d'une fenêtre constante vaut c_j @ sum_k kernel[k] + bias
    def constant(self, c):
        kernel = self.lifting.variables["params"]["kernel"]   # (p+q+1, 3, dv)
        bias   = self.lifting.variables["params"]["bias"]     # (dv,)
        return self._head(c.T @ jnp.sum(kernel, axis=0) + bias)


model = stencil_conv1D(activation=nn.gelu, dv=64)
