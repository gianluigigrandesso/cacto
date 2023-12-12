import sys
import math
import casadi
import numpy as np
import pinocchio.casadi as cpin

class TO_Casadi:
    
    def __init__(self, env, conf, env_TO, w_S=0):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param u_min :                      (float array) Action lower bound array
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param dt :                         (float) Timestep

        :input system_id :                      (str) Id system
        
        :input w_S :                            (float) Sobolev-training weight
        '''
        
        self.env = env
        self.conf = conf

        self.nx = conf.nx
        self.nu = conf.na

        self.w_S = w_S

        self.CAMS = env_TO
    
    def TO_System_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Create and solbe TO casadi problem '''
        ### PROBLEM
        opti = casadi.Opti()

        # The control models are stored as a collection of shooting nodes called running models, with an additional terminal model.
        self.runningSingleModel = self.CAMS('running_model', self.conf)
        runningModels = [ self.runningSingleModel for _ in range(T) ]
        self.terminalModel = self.CAMS('terminal_model', self.conf)
        
        # Decision variables
        xs = [ opti.variable(model.nx) for model in runningModels+[self.terminalModel] ]     # state variable
        us = [ opti.variable(model.nu) for model in runningModels ]                          # control variable
        
        # Roll out loop, summing the integral cost and defining the shooting constraints.
        total_cost = 0
        
        opti.subject_to(xs[0] == ICS_state[:-1])

        for t in range(T):
            x_next, r_cost = runningModels[t].step_fun(xs[t], us[t])
            opti.subject_to(xs[t + 1] == x_next )
            total_cost += r_cost 
        r_cost_final = self.terminalModel.cost(xs[-1], us[-1])
        total_cost += r_cost_final
        
        ### SOLVE
        opti.minimize(total_cost)  
        
        # Create warmstart
        init_x_TO = [np.array(init_TO_states[i,:-1]) for i in range(T+1)]
        init_u_TO = [np.array(init_TO_controls[i,:]) for i in range(T)]

        for x,xg in zip(xs,init_x_TO): opti.set_initial(x,xg)
        for u,ug in zip(us,init_u_TO): opti.set_initial(u,ug)

        # Set solver options
        opts = {'ipopt.linear_solver':'ma57', 'ipopt.sb': 'yes','ipopt.print_level': 0, 'print_time': 0} #, 'ipopt.max_iter': 500} 
        opti.solver("ipopt", opts) 
        
        try:
            opti.solve()
            TO_states = np.array([ opti.value(x) for x in xs ])
            TO_controls = np.array([ opti.value(u) for u in us ])
            TO_total_cost = opti.value(total_cost)
            TO_ee_pos_arr = np.empty((T+1,3))
            TO_step_cost = np.empty(T+1)
            for n in range(T):
                TO_ee_pos_arr[n,:] = np.reshape(runningModels[n].p_ee(TO_states[n,:]),-1)
                TO_step_cost[n] = runningModels[n].cost(TO_states[n,:], TO_controls[n,:])
            TO_ee_pos_arr[-1,:] = np.reshape(self.terminalModel.p_ee(TO_states[-1,:]),-1)
            TO_step_cost[-1] = self.terminalModel.cost(TO_states[-1,:], TO_controls[-1,:])
            success_flag = 1
        except:
            print('ERROR in convergence, returning debug values')
            TO_states = np.array([ opti.debug.value(x) for x in xs ])
            TO_controls = np.array([ opti.debug.value(u) for u in us ])
            TO_total_cost = None
            TO_ee_pos_arr = None
            TO_step_cost = None
            success_flag = 0

        return success_flag, TO_controls, TO_states, TO_ee_pos_arr, TO_total_cost, TO_step_cost
    
    def TO_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Retrieve TO problem solution and compute the value function derviative with respect to the state '''
        success_flag, TO_controls, TO_states, TO_ee_pos_arr, _, TO_step_cost = self.TO_System_Solve(ICS_state, init_TO_states, init_TO_controls, T)
        if success_flag == 0:
            return None, None, success_flag, None, None, None 

        if self.w_S != 0:
            # Compute V gradient w.r.t. x (no computation dV/dt)
            dVdx = self.backward_pass(T+1, TO_states, TO_controls) 
        else:
            dVdx = np.zeros((T+1, self.conf.nb_state))

        # Add the last state component (time)
        TO_states = np.concatenate((TO_states, init_TO_states[0,-1] + np.transpose(self.conf.dt*np.array([range(T+1)]))), axis=1)
            
        return TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, dVdx 

    def backward_pass(self, T, TO_states, TO_controls, mu=1e-9):
        ''' Perform the backward-pass of DDP to obtain the derivatives of the Value function w.r.t x '''
        n = self.conf.nb_state-1
        m = self.conf.nb_action

        X_bar = np.zeros((T, n))
        for i in range(n):
            X_bar[:,i] = [TO_states[t,i] for t in range(T)]

        U_bar = np.zeros((T-1, m))
        for i in range(m):
            U_bar[:,i] = [TO_controls[t,i] for t in range(T-1)]
 
        # The task is defined by a quadratic cost: 
        # sum_{i=0}^T 0.5 x' l_{xx,i} x + l_{x,i} x +  0.5 u' l_{uu,i} u + l_{u,i} u + x' l_{xu,i} u
        l_x  = np.zeros((T, n))
        l_xx = np.zeros((T, n, n))
        l_u  = np.zeros((T-1, m))
        l_uu = np.zeros((T-1, m, m))
        l_xu = np.zeros((T-1, n, m))
        
        # The cost-to-go is defined by a quadratic function: 0.5 x' Q_{xx,i} x + Q_{x,i} x + ...
        Q_xx = np.zeros((T-1, n, n))
        Q_x  = np.zeros((T-1, n))
        Q_uu = np.zeros((T-1, m, m))
        Q_u  = np.zeros((T-1, m))
        Q_xu = np.zeros((T-1, n, m))
        
        x = casadi.SX.sym('x',n,1)
        u = casadi.SX.sym('u',m,1)

        running_cost = -self.runningSingleModel.cost(x, u)
        terminal_cost = -self.terminalModel.cost(x, u)

        running_cost_xx, running_cost_x = casadi.hessian(running_cost,x)
        running_cost_uu, running_cost_u = casadi.hessian(running_cost,u)
        running_cost_xu = casadi.jacobian(casadi.jacobian(running_cost,x),u)
        terminal_cost_xx, terminal_cost_x = casadi.hessian(terminal_cost,x)

        fun_running_cost_x   = casadi.Function('fun_running_cost_x',  [x],  [running_cost_x], ['x'], ['running_cost_x'])
        fun_running_cost_xx  = casadi.Function('fun_running_cost_xx', [x],  [running_cost_xx], ['x'], ['running_cost_xx'])
        fun_running_cost_xu  = casadi.Function('fun_running_cost_xu', [x,u],[running_cost_xu], ['x','u'], ['running_cost_xu'])
        fun_running_cost_u   = casadi.Function('fun_running_cost_u',  [u],  [running_cost_u], ['u'], ['running_cost_u'])
        fun_running_cost_uu  = casadi.Function('fun_running_cost_uu', [u],  [running_cost_uu], ['u'], ['running_cost_uu'])
        fun_terminal_cost_x  = casadi.Function('fun_terminal_cost_x', [x],  [terminal_cost_x], ['x'], ['terminal_cost_x'])
        fun_terminal_cost_xx = casadi.Function('fun_terminal_cost_xx',[x],  [terminal_cost_xx], ['x'], ['terminal_cost_xx'])
        
        # The Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
        V_xx = np.zeros((T, n, n))
        V_x  = np.zeros((T, n+1))

        # Dynamics derivatives w.r.t. x and u
        A = np.zeros((T-1, n, n))
        B = np.zeros((T-1, n, m))
        
        # Initialize value function
        l_x[-1,:], l_xx[-1,:,:] = np.reshape(fun_terminal_cost_x(X_bar[-1,:]),n), fun_terminal_cost_xx(X_bar[-1,:])
        V_xx[T-1,:,:] = l_xx[-1,:,:]
        V_x[T-1,:-1]    = l_x[-1,:]

        for i in range(T-2, -1, -1):
            # Compute dynamics Jacobians
            A[i,:,:], B[i,:,:] = self.env.augmented_derivative(X_bar[i,:], U_bar[i,:])

            # Compute the gradient of the cost function at X=X_bar
            l_x[i,:], l_xx[i,:,:] = np.reshape(fun_running_cost_x(X_bar[i,:]),n), fun_running_cost_xx(X_bar[i,:])
            l_u[i,:],l_uu[i,:,:]  = np.reshape(fun_running_cost_u(U_bar[i,:]),m), fun_running_cost_uu(U_bar[i,:])
            l_xu[i,:,:] = fun_running_cost_xu(X_bar[i,:], U_bar[i,:])                                                            
            
            # Compute regularized cost-to-go
            Q_x[i,:]     = l_x[i,:] + A[i,:,:].T @ V_x[i+1,:-1]
            Q_u[i,:]     = l_u[i,:] + B[i,:,:].T @ V_x[i+1,:-1]
            Q_xx[i,:,:]  = l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            Q_uu[i,:,:]  = l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            Q_xu[i,:,:]  = l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
                
            Qbar_uu       = Q_uu[i,:,:] + mu*np.identity(m)
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu)

            # Compute the derivative of the Value function w.r.t. x                
            V_x[i,:-1]    = Q_x[i,:]  - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_u[i,:]
            V_xx[i,:]   = Q_xx[i,:] - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_xu[i,:,:].T

        return V_x