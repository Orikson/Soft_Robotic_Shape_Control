'''
Kirchhoff rod backbone dynamics. Assumes cylindrical backbone with circular cross-section.
'''

import jax.numpy as np
import jax
import jaxopt
from functools import partial
from typing import NamedTuple

from .util import so3, unpack_sing, unpackh_sing


class Backbone(NamedTuple):
    verbose: bool
    n: int          # number of discrete points PER SEGMENT 
    L: float        # length PER SEGMENT
    ds: float
    r: float
    rho: float
    g: np.ndarray
    dt: float

    A: float
    J: np.ndarray
    Kbt: np.ndarray
    Kse: np.ndarray
    Kbt_inv: np.ndarray
    Kse_inv: np.ndarray

    v: np.ndarray
    v_s: np.ndarray
    u_s: np.ndarray

    Bse: np.ndarray
    Bbt: np.ndarray
    # C: np.ndarray
    
    alpha: float
    c0: float
    c1: float
    c2: float
    d1: float

    Kse_c0Bse: np.ndarray
    Kbt_c0Bbt: np.ndarray

    tendons: tuple
    n_tendons: int
    n_sections: int

def init_bb(n, L, r, E, G, rho, dt, tendons, verbose=False):
    A = np.pi * r**2
    I = np.pi * r**4 / 4
    J = 2 * I

    Kbt = np.diag(np.array([E*I, E*I, G*J]))
    Kse = np.diag(np.array([G*A, G*A, E*A]))

    Bse = np.diag(np.array([0., 0., 0.]))
    Bbt = np.diag(np.array([5e-4, 5e-4, 5e-4])) * 1e-9

    alpha = -0.2
    c0 = (1.5 + alpha) / (dt * (1 + alpha))
    c1 = -2 / dt
    c2 = (0.5 + alpha) / (dt * (1 + alpha))
    d1 = alpha / (1 + alpha)

    self = Backbone(
        verbose,
        n,
        L,
        L / (n - 1),
        r,
        rho,
        np.array([-9.81, 0., 0.]),
        dt,
        A,
        np.diag(np.array([I, I, J])),
        Kbt,
        Kse,
        np.linalg.inv(Kbt),
        np.linalg.inv(Kse),
        np.array([0., 0., 1.]),
        np.array([0., 0., 1.]),
        np.array([0., 0., 0.]),
        Bse,
        Bbt,
        alpha,
        c0,
        c1,
        c2,
        d1,
        Kse + c0 * Bse,
        Kbt + c0 * Bbt,
        # (*tendons, make_tendon(1, np.array([0]), np.array([0]))),
        tendons,
        int(sum([tendon.n_tendons for tendon in tendons])),
        int(len(tendons))
    )
    return self


################
# STATIC FUNCS #
################

@partial(jax.jit, static_argnums=(3))
def _ode(self, x, tau, sec_i):
    '''
    Static ODE for singular x in shape (21)

    Applied at section `sec_i`
    '''
    p, R, u, q, w = unpack_sing(x)

    # Compute tendon forces
    r = self.tendons[sec_i].r

    pb_si = (np.cross(u[:,0], r) + self.v)[:,:,None]
    pb_s_norm = np.linalg.norm(pb_si[:,:,0], axis=1)

    A_i = -so3(pb_si) @ so3(pb_si) * (tau / np.power(pb_s_norm, 3))[:,None,None]
    G_i = -A_i @ so3(r[:,:,None])
    a_i = A_i @ np.cross(u[:,0], pb_si[:,:,0])[:,:,None]
    b_i = np.cross(r, a_i[:,:,0])[:,:,None]
    H_i = so3(r[:,:,None]) @ G_i

    a = np.sum(a_i, axis=0)
    b = np.sum(b_i, axis=0)
    A = np.sum(A_i, axis=0) + self.Kse
    G = np.sum(G_i, axis=0)
    H = np.sum(H_i, axis=0) + self.Kbt
    
    # Compute us
    # nb = self.v - self.v_s
    mb = self.Kbt @ u

    lhs = H
    rhs = -np.cross(u[:,0], mb[:,0]) - b[:,0]
    # rhs = -np.cross(u[:,0], mb[:,0]) - np.cross(self.v, nb) - b[:,0]

    p_s = R @ self.v[:,None]
    R_s = R @ so3(u[None,:])
    u_s = np.linalg.solve(lhs, rhs)[:,None]
    q_s = np.zeros_like(p_s)
    w_s = np.zeros_like(p_s)

    z = np.concatenate((u, np.zeros_like(u), np.zeros_like(u), u_s), axis=0)

    return np.concatenate((p_s, np.reshape(R_s, (9,1)), u_s, q_s, w_s), axis=0), z
        

# @partial(jax.jit, static_argnums=(5,6))
# def _static_sec_euler(self, y0, tau, X, Xh, sec_i, n):
#     # integrate over a section
#     Y = 0*X
#     Z = 0*Xh
    
#     ds = self.ds
    
#     # for i in range(1, n):
#     def compute_next(i, carry):
#         Y, Z, y0 = carry

#         ys1, z1 = _ode(self, y0,                tau, sec_i); ys1 = ys1[:,0]
#         ys2, z2 = _ode(self, y0 + ds * ys1 / 2, tau, sec_i); ys2 = ys2[:,0]
#         ys3, _  = _ode(self, y0 + ds * ys2 / 2, tau, sec_i); ys3 = ys3[:,0]
#         ys4, _  = _ode(self, y0 + ds * ys3,     tau, sec_i); ys4 = ys4[:,0]
        
#         y0 = y0 + self.ds * (ys1 + 2 * ys2 + 2 * ys3 + ys4) / 6
        
#         Y = Y.at[i,:].set(y0)
        
#         Z = Z.at[2*i,:].set(z1[:,0])
#         Z = Z.at[2*i+1,:].set(z2[:,0])

#         return (Y, Z, y0)
    
#     Y, Z, y0 = jax.lax.fori_loop(0, n, compute_next, (Y, Z, y0))
    
#     ys1, z1 = _ode(self, Y[n-1,:],                tau, sec_i); ys1 = ys1[:,0]
#     ___, z2 = _ode(self, Y[n-1,:] + ds * ys1 / 2, tau, sec_i)
#     Z = Z.at[2*n,:].set(z1[:,0])
#     Z = Z.at[2*n+1,:].set(z2[:,0])
    
#     return Y, Z

@partial(jax.jit, static_argnums=(5,6))
def _static_sec_euler(self, y0, tau, X, Xh, sec_i, n):
    # integrate over a section
    Y = 0*X
    Z = 0*Xh
    
    ds = self.ds
    
    # for i in range(1, n):
    def compute_next(i, carry):
        Y, Z, y0 = carry

        ys1, z1 = _ode(self, y0, tau, sec_i); ys1 = ys1[:,0]
        y0 = y0 + self.ds * ys1
        
        Y = Y.at[i,:].set(y0)
        Z = Z.at[i,:].set(z1[:,0])

        return (Y, Z, y0)
    
    Y, Z, y0 = jax.lax.fori_loop(0, n, compute_next, (Y, Z, y0))
    
    _, z1 = _ode(self, Y[n-1,:], tau, sec_i)
    Z = Z.at[n,:].set(z1[:,0])
    
    return Y, Z

@partial(jax.jit, static_argnums=(3,4))
def _static_euler(self, y0, taus, n, n_segments, X, Xh):
    # integrate over backbone
    # n - discretization PER SEGMENT
    # taus - tuple of np.ndarrays for tendon forces. has len `n_segments + 1` where the last tau is [0]

    Y = np.zeros_like(X)
    Z = np.zeros_like(Xh)

    for i in range(n_segments):
    # def comp_next(i, carry):
        # Y, Z, y0 = carry

        # conds for originating tendons
        r = self.tendons[i].r
        pb_si = (np.cross(y0[12:15], r) + self.v)[:,:,None]
        pb_si_norm = np.linalg.norm(pb_si[:,:,0], axis=1)
        F_io = taus[i][:,None,None] * pb_si / pb_si_norm[:,None,None]
        L_io = np.cross(r, F_io[:,:,0])

        # moment due to tendon BCs
        L = np.sum(L_io, axis=0)

        y0 = y0.at[12:15].add(-self.Kbt_inv @ L)

        # solve body
        # _Y, _Z = _static_sec_euler(self, y0, taus[i], X[i*n:(i+1)*n], Xh[2*i*n:2*(i+1)*n+2], i, n)
        _Y, _Z = _static_sec_euler(self, y0, taus[i], X[i*n:(i+1)*n], Xh[i*n:(i+1)*n+1], i, n)
        Y = Y.at[i*n:(i+1)*n].set(_Y)
        # Z = Z.at[2*i*n:2*(i+1)*n+2].set(_Z)
        Z = Z.at[i*n:(i+1)*n+1].set(_Z)

        # add boundary condition to y0
        y0 = Y[(i+1)*n-1]

        # conds for terminating tendons
        r = self.tendons[i].r
        pb_si = (np.cross(y0[12:15], r) + self.v)[:,:,None]
        pb_si_norm = np.linalg.norm(pb_si[:,:,0], axis=1)
        F_it = -taus[i][:,None,None] * pb_si / pb_si_norm[:,None,None]
        L_it = np.cross(r, F_it[:,:,0])

        # moment due to tendon BCs
        L = np.sum(L_it, axis=0)

        y0 = y0.at[12:15].add(-self.Kbt_inv @ L)

        # return (Y, Z, y0)

    # Y, Z, y0 = jax.lax.fori_loop(0, n_segments, comp_next, (Y, Z, y0))

    return Y, Z

@partial(jax.jit, static_argnums=(5,6))
def sta_obj(u0, bb, taus, X, Xh, n, n_segments):
    y0 = np.zeros(21)
    y0 = y0.at[3:12].set(np.eye(3).flatten())
    y0 = y0.at[12:15].set(u0)

    Y, _ = _static_euler(bb, y0, taus, n, n_segments, X, Xh)

    # Tip boundary condition
    uL = Y[-1,12:15]
    uL_t = np.zeros((3))
    mbL = bb.Kbt @ uL[:,None] + bb.Bbt @ uL_t[:,None]

    # cond for terminating tendon
    r = bb.tendons[-1].r
    pb_si = (np.cross(uL, r) + bb.v)[:,:,None]
    pb_si_norm = np.linalg.norm(pb_si[:,:,0], axis=1)
    F_it = -taus[-1][:,None,None] * pb_si / pb_si_norm[:,None,None]
    L_it = np.cross(r, F_it[:,:,0])
    L = np.sum(L_it, axis=0)

    return np.power(mbL[:,0] - L, 2)

@partial(jax.jit, static_argnums=(5,6))
def sta_get(u0, bb, taus, X, Xh, n, n_segments):
    y0 = np.zeros(21)
    y0 = y0.at[3:12].set(np.eye(3).flatten())
    y0 = y0.at[12:15].set(u0)

    Y, Z = _static_euler(bb, y0, taus, n, n_segments, X, Xh)
    return Y, -bb.c0 * Z, Z


#################
# DYNAMIC FUNCS #
#################

@partial(jax.jit, static_argnums=(4))
def _pde(self, x, xh, tau, sec_i):
    '''
    Dynamic ODE for singular x and xh in shape (21) and (12)

    Applied at section `sec_i`
    '''
    p, R, u, q, w = unpack_sing(x)

    # Compute tendon forces
    r = self.tendons[sec_i].r

    pb_si = (np.cross(u[:,0], r) + self.v)[:,:,None]
    pb_s_norm = np.linalg.norm(pb_si[:,:,0], axis=1)

    A_i = -so3(pb_si) @ so3(pb_si) * (tau / np.power(pb_s_norm, 3))[:,None,None]
    G_i = -A_i @ so3(r[:,:,None])
    a_i = A_i @ np.cross(u[:,0], pb_si[:,:,0])[:,:,None]
    b_i = np.cross(r, a_i[:,:,0])[:,:,None]
    H_i = so3(r[:,:,None]) @ G_i

    a = np.sum(a_i, axis=0)
    b = np.sum(b_i, axis=0)
    A = np.sum(A_i, axis=0) + self.Kse_c0Bse
    G = np.sum(G_i, axis=0)
    H = np.sum(H_i, axis=0) + self.Kbt_c0Bbt
    
    # Compute time derivatives
    uh, qh, wh, ush = unpackh_sing(xh)
    # v_t omitted, assumed to be zero
    # vsh omitted, assumed to be zero 
    u_t = self.c0 * u + uh
    # q_t = self.c0 * q + qh
    w_t = self.c0 * w + wh

    # Compute us
    # nb = self.v - self.v_s
    mb = self.Kbt @ u + self.Bbt @ u_t

    lhs = H
    rhs = -(
            np.cross(u[:,0], mb[:,0]) +
            (self.Bbt @ ush)[:,0]
        ) \
        - b[:,0] + \
        self.rho * (
            np.cross(w[:,0], (self.J @ w)[:,0]) + self.J @ w_t
        )[:,0]
    
    p_s = R @ self.v[:,None]
    R_s = R @ so3(u[None,:])
    u_s = np.linalg.solve(lhs, rhs)[:,None]
    q_s = -np.cross(u[:,0], q[:,0])[:,None] + np.cross(w[:,0], self.v)[:,None]
    w_s = u_t - np.cross(u[:,0], w[:,0])[:,None]

    return np.concatenate((p_s, np.reshape(R_s, (9,1)), u_s, q_s, w_s), axis=0), \
        np.concatenate((u, q, w, u_s), axis=0)

# @partial(jax.jit, static_argnums=(5,6))
# def _dynamic_sec_euler(self, y0, tau, X, Xh, sec_i, n):
#     # integrate over a section
#     Y = 0*X
#     Z = 0*Xh
    
#     ds = self.ds
    
#     # for i in range(1, n):
#     def compute_next(i, carry):
#         Y, Z, y0 = carry

#         ys1, z1 = _pde(self, y0,                Xh[2*i,:],   tau, sec_i); ys1 = ys1[:,0]
#         ys2, z2 = _pde(self, y0 + ds * ys1 / 2, Xh[2*i+1,:], tau, sec_i); ys2 = ys2[:,0]
#         ys3, _  = _pde(self, y0 + ds * ys2 / 2, Xh[2*i+1,:], tau, sec_i); ys3 = ys3[:,0]
#         ys4, _  = _pde(self, y0 + ds * ys3,     Xh[2*i+2,:], tau, sec_i); ys4 = ys4[:,0]
        
#         y0 = y0 + self.ds * (ys1 + 2 * ys2 + 2 * ys3 + ys4) / 6
        
#         Y = Y.at[i,:].set(y0)
        
#         Z = Z.at[2*i,:].set(z1[:,0])
#         Z = Z.at[2*i+1,:].set(z2[:,0])

#         return (Y, Z, y0)
    
#     Y, Z, y0 = jax.lax.fori_loop(0, n, compute_next, (Y, Z, y0))
    
#     ys1, z1 = _pde(self, Y[n-1,:],                Xh[2*n,:],   tau, sec_i); ys1 = ys1[:,0]
#     ___, z2 = _pde(self, Y[n-1,:] + ds * ys1 / 2, Xh[2*n+1,:], tau, sec_i)
#     Z = Z.at[2*n,:].set(z1[:,0])
#     Z = Z.at[2*n+1,:].set(z2[:,0])
    
#     return Y, Z

@partial(jax.jit, static_argnums=(5,6))
def _dynamic_sec_euler(self, y0, tau, X, Xh, sec_i, n):
    # integrate over a section
    Y = 0*X
    Z = 0*Xh
    
    ds = self.ds
    
    # for i in range(1, n):
    def compute_next(i, carry):
        Y, Z, y0 = carry

        ys1, z1 = _pde(self, y0, Xh[i,:], tau, sec_i); ys1 = ys1[:,0]
        y0 = y0 + self.ds * ys1
        
        Y = Y.at[i,:].set(y0)        
        Z = Z.at[i,:].set(z1[:,0])

        return (Y, Z, y0)
    
    Y, Z, y0 = jax.lax.fori_loop(0, n, compute_next, (Y, Z, y0))
    
    _, z1 = _pde(self, Y[n-1,:], Xh[n,:], tau, sec_i)
    Z = Z.at[n,:].set(z1[:,0])
    
    return Y, Z


@partial(jax.jit, static_argnums=(3,4))
def _dynamic_euler(self, y0, taus, n, n_segments, X, Xh):
    # integrate over backbone
    # n - discretization PER SEGMENT
    # taus - tuple of np.ndarrays for tendon forces. has len `n_segments + 1` where the last tau is [0]

    Y = np.zeros_like(X)
    Z = np.zeros_like(Xh)

    for i in range(n_segments):
    # def comp_next(i, carry):
        # Y, Z, y0 = carry

        # conds for originating tendons
        r = self.tendons[i].r
        pb_si = (np.cross(y0[12:15], r) + self.v)[:,:,None]
        pb_si_norm = np.linalg.norm(pb_si[:,:,0], axis=1)
        F_io = taus[i][:,None,None] * pb_si / pb_si_norm[:,None,None]
        L_io = np.cross(r, F_io[:,:,0])

        # moment due to tendon BCs
        L = np.sum(L_io, axis=0)

        y0 = y0.at[12:15].add(-self.Kbt_inv @ L)

        # solve body
        # _Y, _Z = _dynamic_sec_euler(self, y0, taus[i], X[i*n:(i+1)*n], Xh[2*i*n:2*(i+1)*n+2], i, n)
        _Y, _Z = _dynamic_sec_euler(self, y0, taus[i], X[i*n:(i+1)*n], Xh[i*n:(i+1)*n+1], i, n)
        Y = Y.at[i*n:(i+1)*n].set(_Y)
        # Z = Z.at[2*i*n:2*(i+1)*n+2].set(_Z)
        Z = Z.at[i*n:(i+1)*n+1].set(_Z)

        # add boundary condition to y0
        y0 = Y[(i+1)*n-1]

        # conds for terminating tendons
        r = self.tendons[i].r
        pb_si = (np.cross(y0[12:15], r) + self.v)[:,:,None]
        pb_si_norm = np.linalg.norm(pb_si[:,:,0], axis=1)
        F_it = -taus[i][:,None,None] * pb_si / pb_si_norm[:,None,None]
        L_it = np.cross(r, F_it[:,:,0])

        # moment due to tendon BCs
        L = np.sum(L_it, axis=0)

        y0 = y0.at[12:15].add(-self.Kbt_inv @ L)

        # return (Y, Z, y0)

    # Y, Z, y0 = jax.lax.fori_loop(0, n_segments, comp_next, (Y, Z, y0))

    return Y, Z

@partial(jax.jit, static_argnums=(5,6))
def dyn_obj(u0, bb, taus, X, Xh, n, n_segments):
    y0 = np.zeros(21)
    y0 = y0.at[3:12].set(np.eye(3).flatten())
    y0 = y0.at[12:15].set(u0)

    Y, _ = _dynamic_euler(bb, y0, taus, n, n_segments, X, Xh)

    # Tip boundary condition
    uL = Y[-1,12:15]
    uL_t = bb.c0 * uL + Xh[-1,0:3]
    mbL = bb.Kbt @ uL[:,None] + bb.Bbt @ uL_t[:,None]

    # cond for terminating tendon
    r = bb.tendons[-1].r
    pb_si = (np.cross(uL, r) + bb.v)[:,:,None]
    pb_si_norm = np.linalg.norm(pb_si[:,:,0], axis=1)
    F_it = -taus[-1][:,None,None] * pb_si / pb_si_norm[:,None,None]
    L_it = np.cross(r, F_it[:,:,0])
    L = np.sum(L_it, axis=0)

    # return np.power(mbL[:,0] - L, 2)
    return np.zeros_like(mbL[:,0])

@partial(jax.jit, static_argnums=(6,7))
def dyn_get(u0, bb, taus, X, Xh, Z_old, n, n_segments):
    y0 = np.zeros(21)
    y0 = y0.at[3:12].set(np.eye(3).flatten())
    y0 = y0.at[12:15].set(u0)

    Y, Z = _dynamic_euler(bb, y0, taus, n, n_segments, X, Xh)
    return Y, (bb.c1 + bb.c0 * bb.d1) * Z + bb.c2 * Z_old + bb.d1 * Xh, Z


##############
# STEP FUNCS #
##############
# sta_rf = jaxopt.Broyden(sta_obj, jit=False, verbose=False)
# dyn_rf = jaxopt.Broyden(dyn_obj, jit=False, verbose=False)

# def init(backbone, taus, X, Xh):
#     u0 = sta_rf.run(np.zeros(3),
#         backbone,
#         taus,
#         X, Xh,
#         backbone.n, backbone.n_sections
#     )

#     X, Xh, Z = sta_get(u0.params, backbone, taus, X, Xh, backbone.n, backbone.n_sections)
#     return X, Xh, Z, u0.params

# def step(backbone, taus, X, Xh, Z, u0):
#     u0 = dyn_rf.run(u0,
#         backbone,
#         taus,
#         X, Xh,
#         backbone.n, backbone.n_sections
#     )

#     X, Xh, Z = dyn_get(u0.params, backbone, taus, X, Xh, Z, backbone.n, backbone.n_sections)
#     return X, Xh, Z, u0.params

def init(backbone, taus, X, Xh):
    X, Xh, Z = sta_get(np.zeros(3), backbone, taus, X, Xh, backbone.n, backbone.n_sections)
    return X, Xh, Z, np.zeros(3)

def step(backbone, taus, X, Xh, Z, u0):
    X, Xh, Z = dyn_get(np.zeros(3), backbone, taus, X, Xh, Z, backbone.n, backbone.n_sections)
    return X, Xh, Z, np.zeros(3)
