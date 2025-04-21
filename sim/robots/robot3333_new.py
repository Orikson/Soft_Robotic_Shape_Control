import jax.numpy as np

from ..tendon import make_tendon
from ..backbone import init_bb

sec_1 = make_tendon(
    np.array([0, 120, 240]), 
    np.array([0.03, 0.03, 0.03])
)
sec_2 = make_tendon(
    np.array([0, 120, 240]), 
    np.array([0.03, 0.03, 0.03])
)
sec_3 = make_tendon(
    np.array([0, 120, 240]), 
    np.array([0.03, 0.03, 0.03])
)
sec_4 = make_tendon(
    np.array([0, 120, 240]), 
    np.array([0.03, 0.03, 0.03])
)

dt = 0.001
backbone = init_bb(
    100,            # n
    0.2,            # L per segment (m)
    0.001,          # r of backbone (m)
    207e9,          # E Young's modulus
    207e9/(2*1.3),  # G
    8000,           # rho density
    dt,             # dt timestep
    (sec_1, sec_2, sec_3, sec_4),
    False
)
