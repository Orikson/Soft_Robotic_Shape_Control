import jax.numpy as np

from ..tendon import make_tendon
from ..backbone import init_bb

sec_1 = make_tendon(
    np.array([0, 120, 240]), 
    np.array([0.05, 0.05, 0.05])
)
sec_2 = make_tendon(
    np.array([0, 120, 240]), 
    np.array([0.05, 0.05, 0.05])
)

dt = 0.00075
backbone = init_bb(
    100,
    1.0,
    0.001,
    207e9,
    207e9/(2*1.3),
    8000,
    dt,
    (sec_1, sec_2),
    False
)
