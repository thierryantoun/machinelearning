import jax
import jax.numpy as jnp
from jax import random
from network_parameters import K, T_target, x


def make_sinus_u0(key):
    k1, k2 = random.split(key)
    a_k = random.uniform(k1, (K,), minval=-1.0, maxval=1.0)
    phase_k = random.uniform(k2, (K,), minval=0.0, maxval=2 * jnp.pi)
    ks = jnp.arange(1, K + 1)
    u = jnp.sum(a_k[:, None] * jnp.sin(2 * jnp.pi * ks[:, None] * x[None, :] + phase_k[:, None]), axis=0)
    return u / jnp.max(jnp.abs(u))


IC_NAMES = ["sinus", "gaussiennes", "polynomes", "constante", "rampe",
            "marches", "creneau", "riemann", "sinus_hf", "champ_gaussien"]
# Probabilités de tirage des types (même ordre que IC_NAMES) : p_i ∝ L_i, loss moyenne d'une
# paire du type i (phys + hf) mesurée par loss_par_type.py sur params_stencil_conv1D.pkl.
# constante a L = 0 (exacte par consistance du flux) -> jamais tirée.
IC_PROBS = jnp.array([0.161, 0.054, 0.016, 0.000, 0.086,
                      0.323, 0.071, 0.083, 0.083, 0.122])
IC_PROBS = IC_PROBS / jnp.sum(IC_PROBS)   # arrondis -> somme exactement 1


def generate_initial_data(key, nb_frequences=K, x=x, ic_type=None):
    key, subkey = random.split(key)
    # 0: sinus, 1: gaussiennes, 2: polynomes, 3: constante, 4: rampe,
    # 5: marches (constante par morceaux, discontinue), 6: creneau (pulse rectangulaire, discontinu)
    # 7: riemann (une seule discontinuite, deux niveaux aleatoires)
    # 8: sinus_hf (ondulation haute frequence k > K sur un fond lisse, sans choc de l'ondulation pendant un bloc)
    # 9: champ_gaussien (champ aléatoire gaussien, longueur de corrélation variée)
    # ic_type imposé (ex. loss_par_type.py) ou tiré selon IC_PROBS
    if ic_type is None:
        ic_type = random.choice(subkey, len(IC_NAMES), p=IC_PROBS)

    key, subkey = random.split(key)

    def make_sinus(subkey):
        """Somme de sinus sur les modes 1..k_max, k_max tiré entre 1 et K : des signaux à grande
        échelle (peu de modes) jusqu'aux plus fins (doc §2, structures de toutes les tailles)."""
        k1, k2, k3 = random.split(subkey, 3)
        k_max   = random.randint(k3, (), 1, K + 1)
        ks      = jnp.arange(1, K + 1)
        a_k     = random.uniform(k1, (K,), minval=-1.0, maxval=1.0) * (ks <= k_max)
        phase_k = random.uniform(k2, (K,), minval=0.0, maxval=2*jnp.pi)
        return jnp.sum(a_k[:, None] * jnp.sin(2*jnp.pi*ks[:, None]*x[None, :] + phase_k[:, None]), axis=0)

    def make_sinus_hf(subkey):
        """Ondulation haute fréquence (k > K) d'amplitude eps sur un fond lisse de grande échelle
        (modes 1..k_max, k_max <= 3, max|fond| = 1). eps est choisi pour que l'ondulation seule ne
        fasse pas de choc pendant un bloc T_target (t* = 1 / (eps 2 pi kc) > T_target) ; après
        normalisation globale (V ~ 1 + eps) elle reste petite, contrairement à un sinus HF seul
        que la normalisation ramènerait à amplitude 1 (chocs immédiats)."""
        k1, k2, k3, k4, k5, k6 = random.split(subkey, 6)
        kc  = random.randint(k1, (), K + 1, x.shape[0] // 2)       # fréquence au-dessus de K
        eps = random.uniform(k2, (), minval=0.3, maxval=0.9) / (2 * jnp.pi * kc * T_target)
        phi = random.uniform(k3, (), minval=0.0, maxval=2 * jnp.pi)
        K_FOND  = 3
        ks      = jnp.arange(1, K_FOND + 1)
        k_max   = random.randint(k4, (), 1, K_FOND + 1)
        a_k     = random.uniform(k5, (K_FOND,), minval=-1.0, maxval=1.0) * (ks <= k_max)
        phase_k = random.uniform(k6, (K_FOND,), minval=0.0, maxval=2 * jnp.pi)
        fond = jnp.sum(a_k[:, None] * jnp.sin(2 * jnp.pi * ks[:, None] * x[None, :] + phase_k[:, None]), axis=0)
        fond = fond / jnp.maximum(jnp.max(jnp.abs(fond)), 1e-12)
        return fond + eps * jnp.sin(2 * jnp.pi * kc * x + phi)

    def make_gaussiennes(subkey):
        """1 à MAX_G bosses gaussiennes périodiques. Largeurs log-uniformes entre 1 cellule et
        0.3 (bosses étroites à très larges, doc §2) ; les bosses inactives ont une amplitude nulle."""
        k1, k2, k3, k4 = random.split(subkey, 4)
        MAX_G   = 6
        N       = x.shape[0]
        n_g     = random.randint(k4, (), 1, MAX_G + 1)
        centers = random.uniform(k1, (MAX_G,), minval=0.0, maxval=1.0)
        widths  = jnp.exp(random.uniform(k2, (MAX_G,), minval=jnp.log(1.0 / N), maxval=jnp.log(0.3)))
        amps    = random.uniform(k3, (MAX_G,), minval=-1.0, maxval=1.0) * (jnp.arange(MAX_G) < n_g)
        d = x[None, :] - centers[:, None]
        d = d - jnp.round(d)                                   # distance périodique
        return jnp.sum(amps[:, None] * jnp.exp(-d**2 / (2 * widths[:, None]**2)), axis=0)

    def make_polynomes(subkey):
        k1, k2 = random.split(subkey)
        degree = 5
        coeffs = random.uniform(k1, (degree+1,), minval=-5.0, maxval=5.0)
        u = jnp.polyval(coeffs, x)
        u = u - (u[-1] - u[0]) * x - u[0]
        return u

    def make_constante(subkey):
        c = random.uniform(subkey, (), minval=-5.0, maxval=5.0)
        return jnp.ones_like(x) * c

    def make_rampe(subkey):
        "Front en tanh, largeur 1/steepness log-uniforme entre 1 cellule et 1/4 du domaine."
        k1, k2, k3, k4 = random.split(subkey, 4)
        N         = x.shape[0]
        center    = random.uniform(k1, (), minval=0.0, maxval=1.0)
        steepness = jnp.exp(random.uniform(k2, (), minval=jnp.log(4.0), maxval=jnp.log(1.0 * N)))
        amp       = random.uniform(k3, (), minval=-1.0, maxval=1.0)
        sign      = jnp.sign(random.uniform(k4, (), minval=-1.0, maxval=1.0))
        d = x - center
        d = d - jnp.round(d)
        return amp * jnp.tanh(sign * steepness * d)

    def make_marches(subkey):
        """Fonction constante par morceaux : discontinuites de type Riemann multiples (periodique).
        Nombre de segments aléatoire (2..MAX_SEG) -> plateaux de toutes les longueurs en cellules.
        Les bords inactifs sont envoyés hors du domaine (2.0) pour garder des formes statiques."""
        k1, k2, k3 = random.split(subkey, 3)
        MAX_SEG = 10
        n_seg  = random.randint(k3, (), 2, MAX_SEG + 1)
        edges  = random.uniform(k1, (MAX_SEG - 1,), minval=0.0, maxval=1.0)
        edges  = jnp.sort(jnp.where(jnp.arange(MAX_SEG - 1) < n_seg - 1, edges, 2.0))
        levels = random.uniform(k2, (MAX_SEG,), minval=-1.0, maxval=1.0)
        seg = jnp.sum(x[:, None] >= edges[None, :], axis=1)  # indice de segment pour chaque x
        return levels[seg]

    def make_creneau(subkey):
        """Pulse rectangulaire : deux discontinuites (double probleme de Riemann).
        Largeur de quelques cellules à presque tout le domaine (périodique)."""
        k1, k2, k3 = random.split(subkey, 3)
        center = random.uniform(k1, (), minval=0.0, maxval=1.0)
        width  = random.uniform(k2, (), minval=0.02, maxval=0.9)
        amp    = random.uniform(k3, (), minval=-1.0, maxval=1.0)
        d = x - center
        d = d - jnp.round(d)
        return amp * (jnp.abs(d) < 0.5 * width).astype(x.dtype)

    def make_riemann(subkey):
        """Probleme de Riemann simple, type tiré explicitement (doc §7, étape 1) :
        choc (uL > uR) ou détente (uL < uR) en x = split, avec un saut d'au moins 0.1.
        Par périodicité, le raccord en x = 0 donne une discontinuité du type opposé."""
        k1, k2, k3, k4 = random.split(subkey, 4)
        split = random.uniform(k1, (), minval=0.0, maxval=1.0)
        hi    = random.uniform(k2, (), minval=-0.9, maxval=1.0)
        lo    = random.uniform(k3, (), minval=-1.0, maxval=hi - 0.1)
        choc  = random.bernoulli(k4)
        levelL = jnp.where(choc, hi, lo)
        levelR = jnp.where(choc, lo, hi)
        return jnp.where(x < split, levelL, levelR)

    def make_champ_gaussien(subkey):
        """Champ aléatoire gaussien périodique de longueur de corrélation l (doc §7, étape 1) :
        coefficients de Fourier gaussiens filtrés par exp(-(2 pi k l)^2 / 4) (covariance
        exp(-d^2 / (2 l^2))). l log-uniforme entre 2 cellules et la moitié du domaine ->
        structures de toutes les tailles en cellules."""
        k1, k2, k3 = random.split(subkey, 3)
        N  = x.shape[0]
        l  = jnp.exp(random.uniform(k1, (), minval=jnp.log(2.0 / N), maxval=jnp.log(0.5)))
        ks = jnp.arange(N // 2 + 1)
        amp = jnp.exp(-(2 * jnp.pi * ks * l) ** 2 / 4).at[0].set(0.0)
        coeffs = amp * (random.normal(k2, ks.shape) + 1j * random.normal(k3, ks.shape))
        return jnp.fft.irfft(coeffs, n=N)

    u = jax.lax.switch(
        ic_type,
        [make_sinus, make_gaussiennes, make_polynomes, make_constante, make_rampe,
         make_marches, make_creneau, make_riemann, make_sinus_hf, make_champ_gaussien],
        subkey,
    )
    # constante (3) n'est pas renormalisee (sinon toujours +-1). sinus_hf (8) l'est :
    # l'ondulation eps est portee par un fond d'amplitude 1, elle reste petite apres normalisation.
    skip_normalize = (ic_type == 3)
    u = jnp.where(skip_normalize, u, u / jnp.max(jnp.abs(u)))

    return u