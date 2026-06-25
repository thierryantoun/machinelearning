import jax
import jax.numpy as jnp
from flax import linen as nn


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
    kmax: int
    activation: callable
    init_fn: callable
    dv: int = 64

    def setup(self):
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.block1 = FNOBlock(kmax=self.kmax, width=self.dv,
                                activation=self.activation,
                                init_fn=self.init_fn)
        self.block2 = FNOBlock(kmax=self.kmax, width=self.dv,
                                activation=self.activation,
                                init_fn=self.init_fn)
        self.block3 = FNOBlock(kmax=self.kmax, width=self.dv,
                                activation=self.activation,
                                init_fn=self.init_fn)
        self.block4 = FNOBlock(kmax=self.kmax, width=self.dv,
                                activation=self.activation,
                                init_fn=self.init_fn)
        self.blocks = [self.block1, self.block2, self.block3, self.block4]
        self.projection = nn.Dense(features=1, use_bias=True)

    def __call__(self, u):                       
        x = u[:, None]                           
        x = self.lifting(x)                      
        for block in self.blocks:
            x = block(x)                         
        x = self.projection(x)                   
        return x[:, 0]                           