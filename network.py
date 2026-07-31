import jax
import jax.numpy as jnp
from flax import linen as nn

from network_parameters import MODEL, K


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


def avg_pool_1d(v, factor=2):
    n, c = v.shape
    assert n % factor == 0, f"length={n} doit être divisible par factor={factor}"
    return jnp.mean(v.reshape(n // factor, factor, c), axis=1)


def upsample_linear_1d(v, factor=2, target_length=None):
    n, c = v.shape
    if target_length is None:
        target_length = n * factor

    x_old = jnp.arange(n) / n                    # endpoint=False : convention FFT standard
    x_new = jnp.arange(target_length) / target_length

    x_ext = jnp.concatenate([x_old, jnp.array([1.0])])
    x_ext_vals = jnp.concatenate([v, v[:1]], axis=0)  # point fantôme = wraparound

    def interp_channel(col):
        return jnp.interp(x_new, x_ext, col)

    return jax.vmap(interp_channel, in_axes=1, out_axes=1)(x_ext_vals)


class GlobalBranch(nn.Module):
    kmax: int
    width: int
    activation: callable
    init_fn: callable

    def setup(self):
        self.W      = nn.Dense(features=self.width)
        self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
        self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))

    def __call__(self, v):
        R        = self.R_real + 1j * self.R_imag
        Wv       = self.W(v)
        v_hat    = jnp.fft.rfft(v, axis=0)[:self.kmax, :]
        RFv      = jax.vmap(multiply_one_mode)(R, v_hat)
        RFv_full = jnp.zeros((v.shape[0] // 2 + 1, self.width), dtype=jnp.complex64)
        RFv_full = RFv_full.at[:self.kmax, :].set(RFv)
        Finverse = jnp.fft.irfft(RFv_full, n=v.shape[0], axis=0)
        return self.activation(Wv + Finverse)


class LocalBranch(nn.Module):
    activation: callable
    width: int
    kernel_size: int

    def setup(self):
        # convolution essaye de trouver la bonne combinaison de stencil
        self.conv1 = nn.Conv(features=self.width,
                              kernel_size=(self.kernel_size,),
                              padding='CIRCULAR')  # CIRCULAR donc padding périodique
        self.conv2 = nn.Conv(features=self.width,
                              kernel_size=(self.kernel_size,),
                              padding='CIRCULAR')
        self.Wl = nn.Dense(features=self.width)

    def __call__(self, v):
        v_pool     = avg_pool_1d(v, factor=2)
        v_conv1    = self.activation(self.conv1(v_pool))
        v_conv2    = self.activation(self.conv2(v_conv1))
        v_upsample = upsample_linear_1d(v_conv2, factor=2, target_length=v.shape[0])
        concat     = jnp.concatenate([v, v_upsample], axis=-1)
        Wconcat    = self.Wl(concat)  # couche dense mélange les différentes infos, après pooling+conv+reset et avant
        return self.activation(Wconcat)


class LGNO(nn.Module):
    activation: callable
    width: int
    kernel_size: int
    kmax: int
    init_fn: callable

    def setup(self):
        self.fno   = GlobalBranch(kmax=self.kmax, width=self.width, activation=self.activation, init_fn=self.init_fn)
        self.local = LocalBranch(activation=self.activation, width=self.width, kernel_size=self.kernel_size)
        self.Ag = nn.Dense(features=self.width)
        self.Al = nn.Dense(features=self.width)
        self.Ml = nn.Dense(features=self.width)

    def __call__(self, v):
        g = self.fno(v)
        l = self.local(v)
        A_g = self.Ag(g)
        A_l = self.Al(l)
        c = A_g * A_l
        concat  = jnp.concatenate([v, g, l, c], axis=-1)
        Mconcat = self.Ml(concat)
        return v + Mconcat


class LGNONet(nn.Module):
    activation: callable
    width: int
    kmax: int
    init_fn: callable
    kernel_size: int = 5
    n_layers: int = 1

    def setup(self):
        self.P = nn.Dense(features=self.width)
        self.layers = [
            LGNO(activation=self.activation, width=self.width,
                 kmax=self.kmax, init_fn=self.init_fn, kernel_size=self.kernel_size)
            for _ in range(self.n_layers)
        ]
        self.Q1 = nn.Dense(features=self.width)
        self.Q2 = nn.Dense(features=1)

    def __call__(self, u):
        h = u[:, None]
        h = self.P(h)
        for layer in self.layers:
            h = layer(h)
        h = self.activation(self.Q1(h))
        f = self.Q2(h)
        return f[:, 0]


if MODEL == "fno":
    model = FNO1D(kmax=K, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64)
else:
    model = LGNONet(activation=nn.gelu, width=64, kernel_size=5, kmax=K,
                     init_fn=nn.initializers.lecun_normal(), n_layers=4)