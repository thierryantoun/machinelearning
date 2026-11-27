import flax.linen as nn
import jax.numpy as jnp

from network_parameters import P, Q


class stencil_conv1D_block(nn.Module):
    width: int
    activation: callable

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)

    def __call__(self, v):
        return v + self.activation(self.W(v))


class stencil_conv1D(nn.Module):
    activation: callable
    dv: int
    n_blocks: int
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
    # activation d'abord : sans elle, lifting puis blocks_0.W (linéaires) se composeraient en une seule matrice.
    # Ici dans _head, commun à __call__ et constant, pour que N - N_bar reste nul sur une fenêtre constante
    def _head(self, x):
        x = self.activation(x)
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

    # W : (n, p+q+1, 3) fenêtres déjà extraites (cellules j-p .. j+q de la face j+1/2), normalisées et masquées
    # (zéros hors du cône local) -> N, (3, n). Même noyau que __call__ : conv VALID de longueur p+q+1 = sum_k W[k] K[k] + b,
    # donc zéros hors cône = noyau tronqué au cône. Écrit en einsum (produit matriciel) et non via nn.Conv :
    # n * batch mini-signaux de longueur p+q+1 -> autotuning cuDNN qui demande des dizaines de Go d'espace de travail
    def fenetres(self, W):
        kernel = self.lifting.variables["params"]["kernel"]   # (p+q+1, 3, dv)
        bias   = self.lifting.variables["params"]["bias"]     # (dv,)
        return self._head(jnp.einsum("nkc,kcd->nd", W, kernel) + bias)

    # c : (3, n), masque : (n, p+q+1) cône local de chaque face -> N sur la fenêtre constante (c_j, ..., c_j) masquée
    # par le même cône : c_j @ sum_{k dans le cône j} kernel[k] + bias. Même masque que fenetres -> N - N_bar = 0 si fenêtre constante.
    # Le cône k = 1-q_loc .. q_loc ne dépend que de q_loc (masque_cone, loss.py) : les noyaux sommés K_cum[q], q = 0..Q,
    # se calculent une fois (sommes cumulées, (Q+1, 3, dv)) au lieu d'une somme sur la fenêtre par face
    def constant_masque(self, c, masque):
        kernel = self.lifting.variables["params"]["kernel"]   # (p+q+1, 3, dv), indice i = k + p
        bias   = self.lifting.variables["params"]["bias"]     # (dv,)
        S = jnp.concatenate([jnp.zeros_like(kernel[:1]), jnp.cumsum(kernel, axis=0)])   # S[i] = sum_{i' < i} kernel[i']
        q = jnp.arange(self.q + 1)
        K_cum = S[self.p + q + 1] - S[jnp.maximum(self.p + 1 - q, 0)]   # (Q+1, 3, dv) : somme sur i = p+1-q .. p+q
        K_cum = K_cum.at[0].set(0.0)                                     # q = 0 : cône vide
        q_loc = jnp.sum(masque, axis=-1) // 2                            # le masque garde 2 q_loc cellules
        return self._head(jnp.sum(c.T[:, :, None] * K_cum[q_loc], axis=1) + bias)


model = stencil_conv1D(activation=nn.gelu, dv=300, n_blocks=3)   # dv >= 3*(P+Q+1) = 156 : plus de directions de la fenêtre invisibles au lifting
