import jax
import jax.numpy as jnp
from flax import linen as nn

@jax.jit
def multiply_one_mode(R_k, v_k):
    # R_k : (dv, dv)
    # v_k : (dv,)
    return R_k @ v_k

class FNOBlock(nn.Module):
    kmax: int
    width: int
    activation: callable
    init_fn: callable

    def setup(self):
      self.W = nn.Dense(features = self.width)
      self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
      self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))

    def __call__(self,v):
      R = self.R_real + 1j * self.R_imag
      Wv = self.W(v)
      v_hat = jnp.fft.rfft(v, axis=0)
      v_hat_tronque = v_hat[:self.kmax,:]
      RFv = jax.vmap(multiply_one_mode)(R, v_hat_tronque)
      RFv_full = jnp.zeros((v.shape[0]//2+1, self.width), dtype=jnp.complex64)
      RFv_full = RFv_full.at[:self.kmax, :].set(RFv)
      Finverse = jnp.fft.irfft(RFv_full, n=v.shape[0], axis=0)
      out = self.activation(Wv + Finverse)

      return out

class FNO1D(nn.Module):
    kmax: int
    activation: callable
    init_fn: callable
    dv: int = 64
    di: int = 1

    def setup(self):
      self.lifting = nn.Dense(features=self.dv)
      self.blocks = [FNOBlock(kmax=self.kmax, width=self.dv,
                            activation=self.activation,
                            init_fn=self.init_fn) for _ in range(4)]
      self.projection = nn.Dense(features=self.di)

    def __call__(self, x):
      x = self.lifting(x)
      for block in self.blocks:
        x = block(x)
      x = self.projection(x)
      return x


