import jax
import jax.numpy as jnp
import flax.linen as nn

from network_parameters import K, P, Q


def multiply_one_mode(R_k, v_k):
    return R_k @ v_k


class FNOBlock(nn.Module):
    width: int
    kmax: int
    activation: callable
    init_fn: callable

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)
        self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
        self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))

    def __call__(self, v):
        R = self.R_real + 1j * self.R_imag
        Wv = self.W(v)
        v_hat = jnp.fft.rfft(v, axis=0)[:self.kmax, :]
        RFv = jax.vmap(multiply_one_mode)(R, v_hat)
        RFv_full = jnp.zeros((v.shape[0] // 2 + 1, self.width), dtype=jnp.complex64)
        RFv_full = RFv_full.at[:self.kmax, :].set(RFv)
        Finverse = jnp.fft.irfft(RFv_full, n=v.shape[0], axis=0)

        return self.activation(Wv + Finverse)


class FNO1D(nn.Module):
    activation: callable
    init_fn: callable
    dv: int
    kmax: int
    n_blocks: int = 4

    def setup(self):
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.blocks = [
            FNOBlock(width=self.dv, kmax=self.kmax, activation=self.activation,
                     init_fn=self.init_fn)
            for _ in range(self.n_blocks)
        ]
        self.projection = nn.Dense(features=P + Q + 1, use_bias=True)

    # w : fenêtres stencil(u0, P, Q), (n, P+Q+1) -> coefficients c_{j,i}, (n, P+Q+1)
    def __call__(self, w):
        x = self.lifting(w)
        for block in self.blocks:
            x = block(x)
        return self.projection(x)


model = FNO1D(kmax=K, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64)
