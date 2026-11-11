import jax
import jax.numpy as jnp
import flax.linen as nn

from network_parameters import MODEL, KMAX


@jax.jit
def multiply_one_mode(R_k, v_k):
    return R_k @ v_k


class FNOBlock(nn.Module):
    kmax: int
    width: int
    activation: callable
    init_fn: callable

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)
        if self.kmax > 0:
            self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
            self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))

    def __call__(self, v):
        Wv = self.W(v)                       # mélange des canaux, ponctuel

        if self.kmax == 0:                   # pas de partie spectrale
            return self.activation(Wv)

        R = self.R_real + 1j * self.R_imag
        v_hat = jnp.fft.rfft(v, axis=0)[:self.kmax, :]
        RFv = jax.vmap(multiply_one_mode)(R, v_hat)
        RFv_full = jnp.zeros((v.shape[0] // 2 + 1, self.width), dtype=jnp.complex64)
        RFv_full = RFv_full.at[:self.kmax, :].set(RFv)
        Finverse = jnp.fft.irfft(RFv_full, n=v.shape[0], axis=0)

        return self.activation(Wv + Finverse)


class FNO1D(nn.Module):
    kmax: int
    activation: callable
    init_fn: callable
    dv: int
    n_blocks: int = 4

    def setup(self):
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.blocks = [
            FNOBlock(kmax=self.kmax, width=self.dv, activation=self.activation,
                     init_fn=self.init_fn)
            for _ in range(self.n_blocks)
        ]
        self.projection = nn.Dense(features=1, use_bias=True)

    def __call__(self, u):
        x = u[:, None]
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x[:, 0]


model = FNO1D(kmax=KMAX, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64)