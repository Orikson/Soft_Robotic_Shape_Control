import casadi as cs
import numpy as np


class KMPC:
    def __init__(self, A, B, C, Q, R, look_ahead, u_max=20, u_min=0, solver='qrqp'):
        '''
        Given a Koopman linear model 
            z_{k+1} = A z_k + B u_k
            x_k = C z_k
        and a reference trajectory
            {r_i, r_{i+1}, ..., r_{i+look_ahead}}
        computes optimal 
            {u_i, u_{i+1}, ..., u_{i+look_ahead}}
        that minimizes
            sum_{j \in [i,i+look_ahead]} (x_j - r_j)^T Q (x_j - r_j) + u_j^T R u_j
        subject to
            0 <= u_i < u_max
        
        This is a quadratic program, solved using cs.Opti('conic')

        Note that x \in R^N, z \in R^n, u \in R^m

        Args:
            A (np.ndarray): matrix in linear model (n x n)
            B (np.ndarray): matrix in linear model (n x m)
            C (np.ndarray): projection matrix (N x n)
            Q (np.ndarray): weight matrix for state error
            R (np.ndarray): weight matrix for control input magnitude
            look_ahead (float): MPC horizon
            u_max (float): maximum value of u
            u_min (float): minimum value of u
        '''

        self.A = A                                      # n x n
        self.B = B                                      # n x m
        self.C = C                                      # N x n
        self.Q = Q                                      # N x N
        self.R = R                                      # m x m
        self.look_ahead = look_ahead
        self.u_max = u_max
        self.u_min = u_min

        n = A.shape[1]
        m = B.shape[1]
        N = C.shape[0]
        self.n = n
        self.m = m
        self.N = N

        ##################
        # Setup optimizer
        if solver in ['qrqp', 'qpoases', 'osqp', 'proxqp']:
            opti = cs.Opti('conic')
        else:
            opti = cs.Opti()

        # Variables
        z_var = opti.variable(n, look_ahead + 1)        # n x (T+1)
        u_var = opti.variable(m, look_ahead)            # m x T
        
        z_int = opti.parameter(n, 1)                    # n x 1
        x_ref = opti.parameter(N, look_ahead + 1)       # N x (T+1)

        # Cost
        cost = 0

        x_comp = C @ z_var                              # (N x n) @ (n x (T+1)) = N x (T+1)
        r_err = (x_comp - x_ref)                        # N x (T+1)
        
        # Dynamics error
        for i in range(1,look_ahead+1):
            cost += r_err[:,i].T @ Q @ r_err[:,i]       # a scalar hopefully
        
        # Control error
        for i in range(1,look_ahead):
            # u_err = u_var[:,i] - u_var[:,i-1]
            u_err = u_var[:,i-1]
            cost += u_err.T @ R @ u_err
        
        # Terminal error
        # cost += 10 * r_err[:,look_ahead].T @ Q @ r_err[:,look_ahead]

        # Dynamics and control constraints
        opti.subject_to(z_var[:,0] == z_int)
        for i in range(look_ahead):
            z_nxt = A @ z_var[:,i] + B @ u_var[:,i]
            opti.subject_to(z_var[:,i+1] == z_nxt)
            
            opti.subject_to(opti.bounded(u_min, u_var[:,i], u_max))
        
        opti.minimize(cost)
        if solver == 'qpoases':
            opti.solver('qpoases', {})
        elif solver == 'qrqp':
            opti.solver('qrqp', 
                {'print_header': False, 'print_info': False, 'print_iter': False}
            )
        elif solver == 'osqp':
            opti.solver('osqp',
                {'verbose': False, 'print_time': False, 'print_problem': False},
                {'verbose': False}
           )
        elif solver == 'proxqp':
            opti.solver('proxqp',
                        {}
           )
        else:
            opti.solver('ipopt')
        
        self.opti_dict = {
            'opti': opti,
            'z_var': z_var,
            'u_var': u_var,
            'z_int': z_int,
            'x_ref': x_ref,
            'cost': cost
        }
    
    def solve(self, ref, z_init, sol=None):
        opti = self.opti_dict['opti']
        z_var = self.opti_dict['z_var']
        u_var = self.opti_dict['u_var']
        z_int = self.opti_dict['z_int']
        x_ref = self.opti_dict['x_ref']

        if sol is not None:
            opti.set_initial(sol.value_variables())

        opti.set_value(z_int, z_init)
        opti.set_value(x_ref, ref)

        try:
            sol = opti.solve()
            z_val, u_val = sol.value(z_var), sol.value(u_var)
            return u_val[:,0], sol
        except RuntimeError as e:
            print(f'MPC solution failed with {opti.return_status()}')
            print(e)
            return np.zeros(self.m), None

