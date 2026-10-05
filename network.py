import jax
import jax.numpy as jnp
import flax.linen as nn

from network_parameters import MODEL, K


@jax.jit
def multiply_one_mode(R_k, v_k):
    return R_k @ v_k


# ---------------------------------------------------------------------------
# Convolution locale à rayon physique
#   (κ_c * v_c)(x) = ∫_{|y|<=r} κ_c(y) v_c(x - y) dy,   κ_c(y) = Σ_j a_{c,j} φ_j(y)
#   φ_j : fonctions chapeau centrées sur J nœuds régulièrement espacés dans [-r, r]
#   Seuls les a_{c,j} sont appris -> même noyau physique quelle que soit la grille N.
# ---------------------------------------------------------------------------
def hat_basis(y, r, J):
    """φ_j(y) pour tous les points y (N,) et tous les nœuds j -> (N, J)."""
    nodes = jnp.linspace(-r, r, J)
    h = 2 * r / (J - 1)
    return jnp.maximum(0.0, 1.0 - jnp.abs(y[:, None] - nodes[None, :]) / h)


class LocalConv(nn.Module):
    radius: float = 0.05   # portée physique r (domaine [0, 1])
    n_basis: int = 12      # J ; garder h = 2r/(J-1) >= 2 Δx de la grille la plus grossière

    @nn.compact
    def __call__(self, v):                                  # v : (N, C)
        N, C = v.shape
        a = self.param("a", nn.initializers.normal(1e-2), (self.n_basis, C))
        dx = 1.0 / N
        n = jnp.arange(N)
        y = jnp.where(n <= N // 2, n, n - N) * dx           # décalages périodiques signés
        w = hat_basis(y, self.radius, self.n_basis) @ a * dx  # noyau échantillonné (N, C), poids de quadrature dx
        # convolution circulaire par canal (depthwise), O(C N log N)
        return jnp.fft.irfft(jnp.fft.rfft(v, axis=0) * jnp.fft.rfft(w, axis=0), n=N, axis=0)


class FNOBlock(nn.Module):
    kmax: int
    width: int
    activation: callable
    init_fn: callable
    radius: float = 0.05
    n_basis: int = 12

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)
        self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
        self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))
        self.local = LocalConv(radius=self.radius, n_basis=self.n_basis)

    def __call__(self, v):
        R = self.R_real + 1j * self.R_imag

        Wv = self.W(v)                       # mélange des canaux, ponctuel
        Lv = self.local(v)                   # translation locale, toutes les échelles

        v_hat = jnp.fft.rfft(v, axis=0)[:self.kmax, :]
        RFv = jax.vmap(multiply_one_mode)(R, v_hat)
        RFv_full = jnp.zeros((v.shape[0] // 2 + 1, self.width), dtype=jnp.complex64)
        RFv_full = RFv_full.at[:self.kmax, :].set(RFv)
        Finverse = jnp.fft.irfft(RFv_full, n=v.shape[0], axis=0)

        return self.activation(Wv + Lv + Finverse)


class FNO1D(nn.Module):
    kmax: int
    activation: callable
    init_fn: callable
    dv: int
    n_blocks: int = 4
    radius: float = 0.05
    n_basis: int = 12

    def setup(self):
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.blocks = [
            FNOBlock(kmax=self.kmax, width=self.dv, activation=self.activation,
                     init_fn=self.init_fn, radius=self.radius, n_basis=self.n_basis)
            for _ in range(self.n_blocks)
        ]
        self.projection = nn.Dense(features=1, use_bias=True)

    def __call__(self, u):
        x = u[:, None]
        # x_hat = jnp.fft.rfft(x, axis=0)[:self.kmax, :]
        # x = jnp.fft.irfft(x_hat, n=x.shape[0], axis=0)
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x[:, 0]


model = FNO1D(kmax=K, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64,
              radius=0.05, n_basis=12)