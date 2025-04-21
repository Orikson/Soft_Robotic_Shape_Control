import jax.numpy as np
import jax


@jax.jit
def unpack(X):
    '''
    Unpacks X into its constituent parts.
    
    Args:
        X (np.array): state vector. In shape (n, 21).

    Returns:
        p (np.array): position vector. In shape (n, 3, 1).
        R (np.array): rotation matrices. In shape (n, 3, 3).
        u (np.array): angular strain. In shape (n, 3, 1).
        q (np.array): linear velocity. In shape (n, 3, 1).
        w (np.array): angular velocity. In shape (n, 3, 1).
    '''
    p = X[:, 0:3][:, :, None]
    R = X[:, 3:12].reshape(-1, 3, 3)
    u = X[:, 12:15][:, :, None]
    q = X[:, 15:18][:, :, None]
    w = X[:, 18:21][:, :, None]
    
    return p, R, u, q, w

@jax.jit
def unpack_sing(X):
    '''
    Unpacks singular X into its constituent parts.
    
    Args:
        X (np.array): state vector. In shape (21).

    Returns:
        p (np.array): position vector. In shape (3, 1).
        R (np.array): rotation matrices. In shape (3, 3).
        u (np.array): angular strain. In shape (3, 1).
        q (np.array): linear velocity. In shape (3, 1).
        w (np.array): angular velocity. In shape (3, 1).
    '''
    p = X[0:3][:, None]
    R = X[3:12].reshape(3, 3)
    u = X[12:15][:, None]
    q = X[15:18][:, None]
    w = X[18:21][:, None]
    
    return p, R, u, q, w

@jax.jit
def unpackh(Xh):
    '''
    Unpacks history vector Xh into its constituent parts.
    
    Args:
        Xh (np.array): state vector. In shape (n, 12).

    Returns:
        uh (np.array): angular strain. In shape (n, 3, 1).
        qh (np.array): linear velocity. In shape (n, 3, 1).
        wh (np.array): angular velocity. In shape (n, 3, 1).
        ush (np.array): arclength dervative of angular strain. In shape (n, 3, 1).
    '''
    uh = Xh[:, 0:3][:, :, None]
    qh = Xh[:, 3:6][:, :, None]
    wh = Xh[:, 6:9][:, :, None]
    ush = Xh[:, 9:12][:, :, None]
    
    return uh, qh, wh, ush

@jax.jit
def unpackh_sing(Xh):
    '''
    Unpacks singular history vector Xh into its constituent parts.
    
    Args:
        Xh (np.array): state vector. In shape (12).

    Returns:
        uh (np.array): angular strain. In shape (3, 1).
        qh (np.array): linear velocity. In shape (3, 1).
        wh (np.array): angular velocity. In shape (3, 1).
        ush (np.array): arclength dervative of angular strain. In shape (3, 1).
    '''
    uh = Xh[0:3][:, None]
    qh = Xh[3:6][:, None]
    wh = Xh[6:9][:, None]
    ush = Xh[9:12][:, None]
    
    return uh, qh, wh, ush

_so3_I = np.eye(3)

@jax.jit
def so3(u):
    '''
    Computes the skew-symmetric matrix of a vector.
    
    Args:
        u (np.array): vector. In shape (n, 3, 1).
    '''
    u1 = np.cross(_so3_I[:,0], u[:,:,0])
    u2 = np.cross(_so3_I[:,1], u[:,:,0])
    u3 = np.cross(_so3_I[:,2], u[:,:,0])
    return np.stack((u1, u2, u3), axis=1)
