import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from network_parameters import K, P, Q, n


def multiply_one_mode(R_k, v_k):
    return R_k @ v_k


class FNOBlock(nn.Module):
    width: int
    kmax: int          # K0 : nb de modes à la résolution d'entraînement n0 (taille de la table R)
    n0: int            # résolution d'entraînement
    activation: callable
    init_fn: callable

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)
        self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
        self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))

    # noyau spectral en fréquence par cellule (§5.1) : R_tab est la table aux xi_i = i/n0,
    # interpolée linéairement en xi_k = k/N avec K = kappa*N/2 modes. A N = n0 on retombe
    # exactement sur la table ; à 2*n0 les modes pairs aussi. Au-delà du dernier point de
    # la table (xi > (K0-1)/n0), on garde la dernière valeur.
    def spectral_kernel(self, N):
        R_tab = self.R_real + 1j * self.R_imag
        K_N = min(int(round(self.kmax * N / self.n0)), N // 2 + 1)
        t   = np.minimum(np.arange(K_N) * self.n0 / N, self.kmax - 1)
        i0  = np.floor(t).astype(np.int32)
        i1  = np.minimum(i0 + 1, self.kmax - 1)
        w   = (t - i0).astype(np.float32)[:, None, None]
        return (1 - w) * R_tab[i0] + w * R_tab[i1]

    def __call__(self, v):
        N = v.shape[0]
        R = self.spectral_kernel(N)                    # (K_N, width, width)
        K_N = R.shape[0]
        Wv = self.W(v)
        v_hat = jnp.fft.rfft(v, axis=0)[:K_N, :]
        RFv = jax.vmap(multiply_one_mode)(R, v_hat)
        RFv_full = jnp.zeros((N // 2 + 1, self.width), dtype=jnp.complex64)
        RFv_full = RFv_full.at[:K_N, :].set(RFv)
        Finverse = jnp.fft.irfft(RFv_full, n=N, axis=0)

        return self.activation(Wv + Finverse)


class FNO1D(nn.Module):
    activation: callable
    init_fn: callable
    dv: int
    kmax: int
    n0: int = n
    n_blocks: int = 4

    def setup(self):
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.blocks = [
            FNOBlock(width=self.dv, kmax=self.kmax, n0=self.n0, activation=self.activation,
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


model = FNO1D(kmax=K, n0=n, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64)
