'''
Objects to represent tendons in the simulation.

Every segment has their own tendon routing object
'''

import jax.numpy as np
from typing import NamedTuple


class Tendon(NamedTuple):
    n_tendons: int
    tendon_angles: np.ndarray
    tendon_displacements: np.ndarray
    r: np.ndarray

def _rot(theta):
    return np.array([np.cos(theta), np.sin(theta), 0])

def make_tendon(tendon_angles, tendon_displacements):
    self = Tendon(
        len(tendon_angles),
        tendon_angles,
        tendon_displacements,
        np.array([tendon_displacements[i] * _rot(np.radians(tendon_angles[i])) for i in range(len(tendon_angles))])
    )
    return self
