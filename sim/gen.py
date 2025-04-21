import jax.numpy as np
import jax
from tqdm import tqdm

from .backbone import init, step

def unpack_tau(backbone, taus):
    _num = np.array([tendon.n_tendons for tendon in backbone.tendons])
    _sum = np.array([0, *np.cumsum(_num)])
    return tuple(taus[_sum[i]:_sum[i+1]] for i in range(len(_num)))

def gen_data(backbone, taus, tau_m, ramp, hold, key=jax.random.key(0), desc=''):
    '''
    Args:
        backbone: Backbone object
        taus: How many unique control inputs to test over
        tau_m: Maximum magnitude of control inputs
        ramp: Time to ramp from one control input to another
        hold: Time to hold a control input after ramping to it
    '''
    X = np.zeros((backbone.n * backbone.n_sections, 21))
    Xh = np.zeros((backbone.n * backbone.n_sections + 1, 12))
    dt = backbone.dt

    # Number of tendons
    n_tendons = int(backbone.n_tendons)

    # Init backbone
    tau = np.zeros(n_tendons)
    X, Xh, Z, u0 = init(backbone, unpack_tau(backbone, tau), X, Xh)

    # Generate taus (+2 to prevent accidental overflow due to floating point roundoff error)
    tauf = tau_m * jax.random.uniform(key, (taus+2, n_tendons))
    tauf = tauf.at[0].set(tau)

    # Simulation setups
    N = int(taus * (ramp + hold) / dt)
    Yf = np.zeros((N+1, *X.shape))
    Yf = Yf.at[0].set(X)
    Uf = np.zeros((N+1, n_tendons))
    Uf = Uf.at[0].set(tau)

    # Simulation loop
    pbar = tqdm(range(1, N+1))
    pbar.set_description(desc)
    for i in pbar:
        t = i*dt

        tau_ind = int(t // (ramp + hold))
        ramping = t - tau_ind * (ramp + hold) < ramp

        if ramping:
            t0 = tau_ind * (ramp + hold)
            t1 = tau_ind * (ramp + hold) + ramp

            tau = tauf[tau_ind] + (tauf[tau_ind+1]-tauf[tau_ind]) / (t1 - t0) * (t - t0)
        else:
            tau = tauf[tau_ind+1]
        
        X, Xh, Z, u0 = step(backbone, unpack_tau(backbone, tau), X, Xh, Z, u0)
        Uf = Uf.at[i].set(tau)
        Yf = Yf.at[i].set(X)

    return Yf, Uf