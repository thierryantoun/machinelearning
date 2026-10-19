import flax.linen as nn
import jax.numpy as jnp
from jax import lax

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
        # taille du noyau - canaux d'entrée - canaux de sortie : (p+q+1, 1, dv)
        self.kernel = self.param("kernel", nn.initializers.lecun_normal(), (self.p + self.q + 1, 1, self.dv))
        self.bias   = self.param("bias", nn.initializers.zeros, (self.dv,))
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

    def __call__(self, u):
        u_pad = jnp.pad(u, (self.p, self.q), mode="wrap")[None, :, None]   # cellules fantomes pour la périodicité de la conv
        # conv bas niveau pour imposer le nbr de cellules fantomes & pour avoir le noyau à dispo
        x = lax.conv_general_dilated(u_pad, self.kernel, window_strides=(1,), padding="VALID",
                                     dimension_numbers=("NWC", "WIO", "NWC"))[0]   # (NWC -> 1,n+p+q,1) -> (WIO -> p+q+1,1,dv) -> (NWC -> 1,n,dv)
        return self._head(x + self.bias)

    # c : (n,) -> N évalué en j sur la fenêtre constante (c_j, ..., c_j), (n,)
    # la convolution d'une fenêtre constante vaut c_j * sum(kernel) + bias
    def constant(self, c):
        x = c[:, None] * jnp.sum(self.kernel, axis=(0, 1)) + self.bias   # (n, dv)
        return self._head(x)


model = FNO1D(activation=nn.gelu, dv=64)
