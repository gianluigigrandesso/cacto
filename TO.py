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
    
#class SingleIntegrator_CAMS:
#    def __init__(self, name, conf):
#        '''
#        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')#

#        :input conf :                           (Configuration file)#

#            :param robot :                      (RobotWrapper instance) 
#            :param cmodel :                     (Casadi-Pinocchio instance)
#            :param cdata :                      (Casadi-Pinocchio model data)
#            :param dt :                         (float) Timestep
#            :param end_effector_frame_id :      (str) Name of EE-frame#

#            # Cost function parameters
#            :param TARGET_STATE :               (float array) Target position
#            :param cost_funct_param             (float array) Cost function scale and offset factors
#            :param soft_max_param :             (float array) Soft parameters array
#            :param obs_param :                  (float array) Obtacle parameters array
#            :param cost_weights_running :       (float array) Running cost weights vector
#            :param cost_weights_terminal :      (float array) Terminal cost weights vector
#        '''
#        self.name = name
#        self.conf = conf
#        
#        self.nx = self.conf.nx
#        self.nu = self.conf.na#

#        # Rename reward parameters
#        self.offset = self.conf.cost_funct_param[0]
#        self.scale = self.conf.cost_funct_param[1]#

#        self.alpha = self.conf.soft_max_param[0]
#        self.alpha2 = self.conf.soft_max_param[1]#

#        self.XC1 = self.conf.obs_param[0]
#        self.YC1 = self.conf.obs_param[1]
#        self.XC2 = self.conf.obs_param[2]
#        self.YC2 = self.conf.obs_param[3]
#        self.XC3 = self.conf.obs_param[4]
#        self.YC3 = self.conf.obs_param[5]
#        self.A1 = self.conf.obs_param[6]
#        self.B1 = self.conf.obs_param[7]
#        self.A2 = self.conf.obs_param[8]
#        self.B2 = self.conf.obs_param[9]
#        self.A3 = self.conf.obs_param[10]
#        self.B3 = self.conf.obs_param[11]#

#        self.x_des = self.conf.TARGET_STATE[0]
#        self.y_des = self.conf.TARGET_STATE[1]#

#        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
#        cx = casadi.SX.sym("x",self.nx,1)
#        cu = casadi.SX.sym("u",self.nu,1)#

#        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])#

#        self.p_ee = casadi.Function('p_ee', [cx], [self.get_end_effector_position_fun(cx)])#

#        if self.name == 'running_model':
#            self.weights = np.copy(self.conf.cost_weights_running)
#        elif self.name == 'terminal_model':
#            self.weights = np.copy(self.conf.cost_weights_terminal)
#        else:
#            print("The model can be either 'running_model' or 'terminal_model'")
#            import sys
#            sys.exit()#

#        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])#

#        #self.step = casadi.Function('step', [cx,cu], [self.step_fun(cx,cu)])#

#    def get_end_effector_position_fun(self, cx):        
#        return casadi.vertcat(cx[0],cx[1],0)
#    
#    def bound_control_cost(self, action):
#        u_cost = 0
#        for i in range(self.conf.nb_action):
#            u_cost += action[i]*action[i] + 1e2*(np.exp(-self.conf.w_b*(action[i]-self.conf.u_min[i])) + np.exp(-self.conf.w_b*(self.conf.u_max[i]-action[i])))#

#        return u_cost
#    
#    def cost_fun(self, x, u):
#        ''' Compute cost '''
#        p_ee = self.p_ee(x)#

#        ### Penalties representing the obstacle ###
#        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
#        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
#        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)#

#        ### Control effort term ###
#        u_cost = self.bound_control_cost(u)#

#        ### Distence to target term (quadratic term) ###
#        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2#

#        ### Distence to target term (log valley centered at target) ###      
#        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
#        
#        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
# 
#        return cost
#    
#    def simulate_fun(self, x, u):
#        ''' Integrate dynamics '''
#        x_next = casadi.vertcat(x[0] + self.conf.dt*u[0],
#                                x[1] + self.conf.dt*u[1])
#        
#        return x_next
#    
#    def step_fun(self, x, u):
#        ''' Return next state and cost '''
#        cost = self.cost(x, u)
#        
#        x_next = self.x_next(x,u)
#        
#        return x_next, cost#

#class DoubleIntegrator_CAMS:
#    def __init__(self, name, conf):
#        '''
#        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')#

#        :input conf :                           (Configuration file)#

#            :param robot :                      (RobotWrapper instance) 
#            :param cmodel :                     (Casadi-Pinocchio instance)
#            :param cdata :                      (Casadi-Pinocchio model data)
#            :param dt :                         (float) Timestep
#            :param end_effector_frame_id :      (str) Name of EE-frame#

#            # Cost function parameters
#            :param TARGET_STATE :               (float array) Target position
#            :param cost_funct_param             (float array) Cost function scale and offset factors
#            :param soft_max_param :             (float array) Soft parameters array
#            :param obs_param :                  (float array) Obtacle parameters array
#            :param cost_weights_running :       (float array) Running cost weights vector
#            :param cost_weights_terminal :      (float array) Terminal cost weights vector
#        '''
#        self.name = name
#        self.conf = conf
#        
#        self.nq = self.conf.cmodel.nq
#        self.nv = self.conf.cmodel.nv
#        self.nx = self.nq+self.nv
#        self.nu = self.conf.cmodel.nv#

#        # Rename reward parameters
#        self.offset = self.conf.cost_funct_param[0]
#        self.scale = self.conf.cost_funct_param[1]#

#        self.alpha = self.conf.soft_max_param[0]
#        self.alpha2 = self.conf.soft_max_param[1]#

#        self.XC1 = self.conf.obs_param[0]
#        self.YC1 = self.conf.obs_param[1]
#        self.XC2 = self.conf.obs_param[2]
#        self.YC2 = self.conf.obs_param[3]
#        self.XC3 = self.conf.obs_param[4]
#        self.YC3 = self.conf.obs_param[5]
#        self.A1 = self.conf.obs_param[6]
#        self.B1 = self.conf.obs_param[7]
#        self.A2 = self.conf.obs_param[8]
#        self.B2 = self.conf.obs_param[9]
#        self.A3 = self.conf.obs_param[10]
#        self.B3 = self.conf.obs_param[11]#

#        self.x_des = self.conf.TARGET_STATE[0]
#        self.y_des = self.conf.TARGET_STATE[1]#

#        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
#        cx = casadi.SX.sym("x",self.nx,1)
#        cu = casadi.SX.sym("u",self.nu,1)#

#        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])#

#        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cx[:self.nq])
#        self.p_ee = casadi.Function('p_ee', [cx], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])#

#        if self.name == 'running_model':
#            self.weights = np.copy(self.conf.cost_weights_running)
#        elif self.name == 'terminal_model':
#            self.weights = np.copy(self.conf.cost_weights_terminal)
#        else:
#            print("The model can be either 'running_model' or 'terminal_model'")
#            import sys
#            sys.exit()#

#        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])#

#        #self.step = casadi.Function('step', [cx,cu], [self.step_fun(cx,cu)])#

#    def bound_control_cost(self, action):
#        u_cost = 0
#        for i in range(self.conf.nb_action):
#            u_cost += action[i]*action[i] + 1e2*(np.exp(-self.conf.w_b*(action[i]-self.conf.u_min[i])) + np.exp(-self.conf.w_b*(self.conf.u_max[i]-action[i])))#

#        return u_cost#

#    def cost_fun(self, x, u):
#        ''' Compute cost '''
#        p_ee = self.p_ee(x)#

#        ### Penalties representing the obstacle ###
#        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
#        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
#        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)#

#        ### Control effort term ###
#        u_cost = self.bound_control_cost(u)#

#        ### Distence to target term (quadratic term) ###
#        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2#

#        ### Distence to target term (log valley centered at target) ###      
#        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
#        
#        ### Terminal cost on final velocity ###
#        v_cost = 0
#        for i in range(self.nv):
#            v_cost += x[i+self.nq]**2
#        
#        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
# 
#        return cost
#    
#    def simulate_fun(self, x, u):
#        ''' Integrate dynamics '''
#        a = cpin.aba(self.conf.cmodel,self.conf.cdata,x[:self.nq],x[self.nq:],u)
#        
#        if self.conf.integration_scheme == 'SI-Euler':
#            F = casadi.vertcat(x[self.nq:]+self.dt*a, a)
#        elif self.conf.integration_scheme == 'E-Euler':
#            F = casadi.vertcat(x[self.nq:], a)
#        x_next = x + self.conf.dt*F
#        
#        return x_next
#    
#    def step_fun(self, x, u):
#        ''' Return next state and cost '''
#        cost = self.cost(x, u)
#        
#        x_next = self.x_next(x,u)
#        
#        return x_next, cost
#    
#class Car_CAMS:
#    def __init__(self, name, conf):
#        '''
#        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')#

#        :input conf :                           (Configuration file)#

#            :param robot :                      (RobotWrapper instance) 
#            :param cmodel :                     (Casadi-Pinocchio instance)
#            :param cdata :                      (Casadi-Pinocchio model data)
#            :param dt :                         (float) Timestep
#            :param end_effector_frame_id :      (str) Name of EE-frame#

#            # Cost function parameters
#            :param TARGET_STATE :               (float array) Target position
#            :param cost_funct_param             (float array) Cost function scale and offset factors
#            :param soft_max_param :             (float array) Soft parameters array
#            :param obs_param :                  (float array) Obtacle parameters array
#            :param cost_weights_running :       (float array) Running cost weights vector
#            :param cost_weights_terminal :      (float array) Terminal cost weights vector
#        '''
#        self.name = name
#        self.conf = conf
#        
#        self.nx = self.conf.nx
#        self.nu = self.conf.na#

#        # Rename reward parameters
#        self.offset = self.conf.cost_funct_param[0]
#        self.scale = self.conf.cost_funct_param[1]#

#        self.alpha = self.conf.soft_max_param[0]
#        self.alpha2 = self.conf.soft_max_param[1]#

#        self.XC1 = self.conf.obs_param[0]
#        self.YC1 = self.conf.obs_param[1]
#        self.XC2 = self.conf.obs_param[2]
#        self.YC2 = self.conf.obs_param[3]
#        self.XC3 = self.conf.obs_param[4]
#        self.YC3 = self.conf.obs_param[5]
#        self.A1 = self.conf.obs_param[6]
#        self.B1 = self.conf.obs_param[7]
#        self.A2 = self.conf.obs_param[8]
#        self.B2 = self.conf.obs_param[9]
#        self.A3 = self.conf.obs_param[10]
#        self.B3 = self.conf.obs_param[11]#

#        self.x_des = self.conf.TARGET_STATE[0]
#        self.y_des = self.conf.TARGET_STATE[1]#

#        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
#        cx = casadi.SX.sym("x",self.nx,1)
#        cu = casadi.SX.sym("u",self.nu,1)#

#        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])#

#        self.p_ee = casadi.Function('p_ee', [cx], [self.get_end_effector_position_fun(cx)])#

#        if self.name == 'running_model':
#            self.weights = np.copy(self.conf.cost_weights_running)
#        elif self.name == 'terminal_model':
#            self.weights = np.copy(self.conf.cost_weights_terminal)
#        else:
#            print("The model can be either 'running_model' or 'terminal_model'")
#            import sys
#            sys.exit()#

#        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])#

#        #self.step = casadi.Function('step', [cx,cu], [self.step_fun(cx,cu)])#

#    def get_end_effector_position_fun(self, cx):        
#        return casadi.vertcat(cx[0],cx[1],0)
#    
#    def bound_control_cost(self, action):
#        u_cost = 0
#        for i in range(self.conf.nb_action):
#            u_cost += action[i]*action[i] + 1e2*(np.exp(-self.conf.w_b*(action[i]-self.conf.u_min[i])) + np.exp(-self.conf.w_b*(self.conf.u_max[i]-action[i])))#

#        return u_cost
#    
#    def cost_fun(self, x, u):
#        ''' Compute cost '''
#        p_ee = self.p_ee(x)#

#        ### Penalties representing the obstacle ###
#        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
#        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
#        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)#

#        ### Control effort term ###
#        u_cost = self.bound_control_cost(u)#

#        ### Distence to target term (quadratic term) ###
#        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2#

#        ### Distence to target term (log valley centered at target) ###      
#        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
#        
#        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
# 
#        return cost
#    
#    def simulate_fun(self, x, u):
#        ''' Integrate dynamics '''
#        x_next = casadi.vertcat(x[0] + self.conf.dt*x[3]*casadi.cos(x[2]) + self.conf.dt**2*x[4]*casadi.cos(x[2])/2,
#                                x[1] + self.conf.dt*x[3]*casadi.sin(x[2]) + self.conf.dt**2*x[4]*casadi.sin(x[2])/2,
#                                x[2] + self.conf.dt*u[0],
#                                x[3] + self.conf.dt*x[4],
#                                x[4] + self.conf.dt*u[1])
#        
#        return x_next
#    
#    def step_fun(self, x, u):
#        ''' Return next state and cost '''
#        cost = self.cost(x, u)
#        
#        x_next = self.x_next(x,u)
#        
#        return x_next, cost#

#class CarPark_CAMS:
#    def __init__(self, name, conf):
#        '''
#        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')#

#        :input conf :                           (Configuration file)#

#            :param robot :                      (RobotWrapper instance) 
#            :param cmodel :                     (Casadi-Pinocchio instance)
#            :param cdata :                      (Casadi-Pinocchio model data)
#            :param dt :                         (float) Timestep
#            :param end_effector_frame_id :      (str) Name of EE-frame#

#            # Cost function parameters
#            :param TARGET_STATE :               (float array) Target position
#            :param cost_funct_param             (float array) Cost function scale and offset factors
#            :param soft_max_param :             (float array) Soft parameters array
#            :param obs_param :                  (float array) Obtacle parameters array
#            :param cost_weights_running :       (float array) Running cost weights vector
#            :param cost_weights_terminal :      (float array) Terminal cost weights vector
#        '''
#        self.name = name
#        self.conf = conf
#        
#        self.nx = self.conf.nx
#        self.nu = self.conf.na#

#        # Rename reward parameters
#        self.offset = self.conf.cost_funct_param[0]
#        self.scale = self.conf.cost_funct_param[1]#

#        self.alpha = self.conf.soft_max_param[0]
#        self.alpha2 = self.conf.soft_max_param[1]#

#        self.XC1 = self.conf.obs_param[0]
#        self.YC1 = self.conf.obs_param[1]
#        self.XC2 = self.conf.obs_param[2]
#        self.YC2 = self.conf.obs_param[3]
#        self.XC3 = self.conf.obs_param[4]
#        self.YC3 = self.conf.obs_param[5]
#        self.A1 = self.conf.obs_param[6]
#        self.B1 = self.conf.obs_param[7]
#        self.A2 = self.conf.obs_param[8]
#        self.B2 = self.conf.obs_param[9]
#        self.A3 = self.conf.obs_param[10]
#        self.B3 = self.conf.obs_param[11]#

#        self.x_des = self.conf.TARGET_STATE[0]
#        self.y_des = self.conf.TARGET_STATE[1]
#        #self.theta_des = self.conf.TARGET_STATE[2]#

#        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
#        cx = casadi.SX.sym("x",self.nx,1)
#        cu = casadi.SX.sym("u",self.nu,1)#

#        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])#

#        self.p_ee = casadi.Function('p_ee', [cx], [self.get_end_effector_position_fun(cx)])#

#        if self.name == 'running_model':
#            self.weights = np.copy(self.conf.cost_weights_running)
#        elif self.name == 'terminal_model':
#            self.weights = np.copy(self.conf.cost_weights_terminal)
#        else:
#            print("The model can be either 'running_model' or 'terminal_model'")
#            import sys
#            sys.exit()#

#        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])#

#        #self.step = casadi.Function('step', [cx,cu], [self.step_fun(cx,cu)])#

#    def get_end_effector_position_fun(self, cx):      
#        p_ee_tmp = cx[:2] + np.array([[np.cos(cx[2]), -np.sin(cx[2])], [np.sin(cx[2]), np.cos(cx[2])]]).dot(np.array([self.conf.L_delta/2,0])) 
#        return casadi.vertcat(p_ee_tmp[0],p_ee_tmp[1],0)
#    
#    def obs_cost_fun(self, x, y, x_clip, y_clip, Wx, Wy, fv=1, k=50):#

#        obs_cost = (4 + 4 * (y - y_clip + Wy / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * fv * (-np.sqrt(4 + 4 * (y - y_clip - Wy / 2) ** 2 * k ** 2) / 2 + (y - y_clip - Wy / 2) * k) * (4 + 4 * (x - x_clip + Wx / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * (4 + 4 * (y - y_clip - Wy / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * (np.sqrt(4 + 4 * (y - y_clip + Wy / 2) ** 2 * k ** 2) / 2 + (y - y_clip + Wy / 2) * k) * (4 + 4 * (x - x_clip - Wx / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * (np.sqrt(4 + 4 * (x - x_clip + Wx / 2) ** 2 * k ** 2) / 2 + (x - x_clip + Wx / 2) * k) * (-np.sqrt(4 + 4 * (x - x_clip - Wx / 2) ** 2 * k ** 2) / 2 + (x - x_clip - Wx / 2) * k)#

#        return obs_cost
#    
#    def bound_state_fun(self, x, x_step, w_clip, ov=1, mv=0, k=50):
#        #ov=max_pen (3)
#        #w_clip = pi/3
#        #x_clip=0
#        #k=50#

#        cg2 = (mv - ov) * (1 + 2 * (x - x_step + w_clip / 2) * k * (k ** 2 * w_clip ** 2 + 4 * k ** 2 * w_clip * x - 4 * k ** 2 * w_clip * x_step + 4 * x ** 2 * k ** 2 - 8 * k ** 2 * x * x_step + 4 * k ** 2 * x_step ** 2 + 4) ** (-0.1e1 / 0.2e1)) * (1 - 2 * (x - x_step - w_clip / 2) * k * (k ** 2 * w_clip ** 2 - 4 * k ** 2 * w_clip * x + 4 * k ** 2 * w_clip * x_step + 4 * x ** 2 * k ** 2 - 8 * k ** 2 * x * x_step + 4 * k ** 2 * x_step ** 2 + 4) ** (-0.1e1 / 0.2e1)) / 4 + ov       #+ np.log(np.exp(50*((((x-x_step)**2)/((self.conf.delta_bound/2)**2) - 1.0))) + 1)/50#

#        return cg2
#    
#    def bound_control_cost(self, action):
#        u_cost = 0
#        for i in range(self.conf.nb_action):
#            u_cost += action[i]*action[i] + 1e2*(np.exp(-self.conf.w_b*(action[i]-self.conf.u_min[i])) + np.exp(-self.conf.w_b*(self.conf.u_max[i]-action[i])))#

#        return u_cost
#    
#    def cost_fun(self, x, u):
#        ''' Compute cost '''
#        p_ee = self.p_ee(x)
#        theta_ee = x[2]#

#        ### Penalties representing the obstacle ###
#        obs_cost = 0
#        for i in range(len(self.conf.check_points_BF)):
#            check_points_WF_i = np.array([[np.cos(theta_ee), -np.sin(theta_ee)], [np.sin(theta_ee), np.cos(theta_ee)]]).dot(self.conf.check_points_BF[i,:]) + p_ee
#            obs_cost += self.obs_cost_fun(check_points_WF_i[0],check_points_WF_i[1],self.XC1,self.YC1,self.A1,self.B1)
#            obs_cost += self.obs_cost_fun(check_points_WF_i[0],check_points_WF_i[1],self.XC2,self.YC2,self.A2,self.B2)
#            obs_cost += self.obs_cost_fun(check_points_WF_i[0],check_points_WF_i[1],self.XC3,self.YC3,self.A3,self.B3)
#        
#        bound_delta_cost = (x[4]**2-self.conf.delta_bound/2 + ((x[4]**2 - self.conf.delta_bound/2)**2 + 1e-3)**0.5)/100 + self.bound_state_fun(x[4], 0, self.conf.delta_bound) ###self.bound_state_fun(x[4], 0, self.conf.delta_bound) + ((x[4]-0)/10)**2
#        
#        ### Control effort term ###
#        u_cost = self.bound_control_cost(u)#

#        ### Distence to target term (quadratic term) ###
#        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2#

#        ### Distence to target term (log valley centered at target) ###      
#        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 2*0.1 - 2*np.sqrt(0.1))) + 1)/self.alpha2
#        
#        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*x[3]**2 + self.weights[3]*obs_cost + self.weights[6]*u_cost + self.weights[7]*bound_delta_cost - self.offset)
# 
#        return cost
#    
#    def simulate_fun(self, x, u):
#        ''' Integrate dynamics '''
#        x_next = casadi.vertcat(x[0] + self.conf.dt*x[3]*casadi.cos(x[2]),
#                                x[1] + self.conf.dt*x[3]*casadi.sin(x[2]),
#                                x[2] + self.conf.dt*x[3]*casadi.tan(x[4])/self.conf.L_delta,
#                                x[3] + self.conf.dt*u[0],
#                                x[4] + self.conf.dt*u[1]/self.conf.tau_delta)
#        
#        return x_next
#    
#    def step_fun(self, x, u):
#        ''' Return next state and cost '''
#        cost = self.cost(x, u)
#        
#        x_next = self.x_next(x,u)
#        
#        return x_next, cost#

#class Manipulator_CAMS:
#    def __init__(self, name, conf):
#        '''
#        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')#

#        :input conf :                           (Configuration file)#

#            :param robot :                      (RobotWrapper instance) 
#            :param cmodel :                     (Casadi-Pinocchio instance)
#            :param cdata :                      (Casadi-Pinocchio model data)
#            :param dt :                         (float) Timestep
#            :param end_effector_frame_id :      (str) Name of EE-frame#

#            # Cost function parameters
#            :param TARGET_STATE :               (float array) Target position
#            :param cost_funct_param             (float array) Cost function scale and offset factors
#            :param soft_max_param :             (float array) Soft parameters array
#            :param obs_param :                  (float array) Obtacle parameters array
#            :param cost_weights_running :       (float array) Running cost weights vector
#            :param cost_weights_terminal :      (float array) Terminal cost weights vector
#        '''
#        self.name = name
#        self.conf = conf
#        
#        self.nq = self.conf.cmodel.nq
#        self.nv = self.conf.cmodel.nv
#        self.nx = self.nq+self.nv
#        self.nu = self.conf.cmodel.nv#

#        # Rename reward parameters
#        self.offset = self.conf.cost_funct_param[0]
#        self.scale = self.conf.cost_funct_param[1]#

#        self.alpha = self.conf.soft_max_param[0]
#        self.alpha2 = self.conf.soft_max_param[1]#

#        self.XC1 = self.conf.obs_param[0]
#        self.YC1 = self.conf.obs_param[1]
#        self.XC2 = self.conf.obs_param[2]
#        self.YC2 = self.conf.obs_param[3]
#        self.XC3 = self.conf.obs_param[4]
#        self.YC3 = self.conf.obs_param[5]
#        self.A1 = self.conf.obs_param[6]
#        self.B1 = self.conf.obs_param[7]
#        self.A2 = self.conf.obs_param[8]
#        self.B2 = self.conf.obs_param[9]
#        self.A3 = self.conf.obs_param[10]
#        self.B3 = self.conf.obs_param[11]#

#        self.x_des = self.conf.TARGET_STATE[0]
#        self.y_des = self.conf.TARGET_STATE[1]#

#        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
#        cx = casadi.SX.sym("x",self.nx,1)
#        cu = casadi.SX.sym("u",self.nu,1)#

#        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])#

#        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cx[:self.nq])
#        self.p_ee = casadi.Function('p_ee', [cx], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])#

#        if self.name == 'running_model':
#            self.weights = np.copy(self.conf.cost_weights_running)
#        elif self.name == 'terminal_model':
#            self.weights = np.copy(self.conf.cost_weights_terminal)
#        else:
#            print("The model can be either 'running_model' or 'terminal_model'")
#            import sys
#            sys.exit()#

#        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])#

#        #self.step = casadi.Function('step', [cx,cu], [self.step_fun(cx,cu)])#

#    def bound_control_cost(self, action):
#        u_cost = 0
#        for i in range(self.conf.nb_action):
#            u_cost += action[i]*action[i] + 1e2*(np.exp(-self.conf.w_b*(action[i]-self.conf.u_min[i])) + np.exp(-self.conf.w_b*(self.conf.u_max[i]-action[i])))#

#        return u_cost
#    
#    def cost_fun(self, x, u):
#        ''' Compute cost '''
#        p_ee = self.p_ee(x)#

#        ### Penalties representing the obstacle ###
#        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
#        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
#        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)#

#        ### Control effort term ###
#        u_cost = self.bound_control_cost(u)#

#        ### Distence to target term (quadratic term) ###
#        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2#

#        ### Distence to target term (log valley centered at target) ###      
#        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
#        
#        ### Terminal cost on final velocity ###
#        v_cost = 0
#        for i in range(self.nv):
#            v_cost += x[i+self.nq]**2
#        
#        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
# 
#        return cost
#    
#    def simulate_fun(self, x, u):
#        ''' Integrate dynamics '''
#        a = cpin.aba(self.conf.cmodel,self.conf.cdata,x[:self.nq],x[self.nq:],u)
#        
#        if self.conf.integration_scheme == 'SI-Euler':
#            F = casadi.vertcat(x[self.nq:]+self.dt*a, a)
#        elif self.conf.integration_scheme == 'E-Euler':
#            F = casadi.vertcat(x[self.nq:], a)
#        x_next = x + self.conf.dt*F
#        
#        return x_next
#    
#    def step_fun(self, x, u):
#        ''' Return next state and cost '''
#        cost = self.cost(x, u)
#        
#        x_next = self.x_next(x,u)
#        
#        return x_next, cost#

#class UR5_CAMS:
#    def __init__(self, name, conf):
#        '''
#        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')#

#        :input conf :                           (Configuration file)#

#            :param robot :                      (RobotWrapper instance) 
#            :param cmodel :                     (Casadi-Pinocchio instance)
#            :param cdata :                      (Casadi-Pinocchio model data)
#            :param dt :                         (float) Timestep
#            :param end_effector_frame_id :      (str) Name of EE-frame#

#            # Cost function parameters
#            :param TARGET_STATE :               (float array) Target position
#            :param cost_funct_param             (float array) Cost function scale and offset factors
#            :param soft_max_param :             (float array) Soft parameters array
#            :param obs_param :                  (float array) Obtacle parameters array
#            :param cost_weights_running :       (float array) Running cost weights vector
#            :param cost_weights_terminal :      (float array) Terminal cost weights vector
#        '''
#        self.name = name
#        self.conf = conf
#        
#        self.nq = self.conf.cmodel.nq
#        self.nv = self.conf.cmodel.nv
#        self.nx = self.nq+self.nv
#        self.nu = self.conf.cmodel.nv#

#        # Rename reward parameters
#        self.offset = self.conf.cost_funct_param[0]
#        self.scale = self.conf.cost_funct_param[1]#

#        self.alpha = self.conf.soft_max_param[0]
#        self.alpha2 = self.conf.soft_max_param[1]#

#        self.XC1 = self.conf.obs_param[0]
#        self.YC1 = self.conf.obs_param[1]
#        self.ZC1 = self.conf.obs_param[2]
#        self.XC2 = self.conf.obs_param[3]
#        self.YC2 = self.conf.obs_param[4]
#        self.ZC2 = self.conf.obs_param[5]
#        self.XC3 = self.conf.obs_param[6]
#        self.YC3 = self.conf.obs_param[7]
#        self.ZC3 = self.conf.obs_param[8]
#        
#        self.A1 = self.conf.obs_param[9]
#        self.B1 = self.conf.obs_param[10]
#        self.C1 = self.conf.obs_param[11]
#        self.A2 = self.conf.obs_param[12]
#        self.B2 = self.conf.obs_param[13]
#        self.C2 = self.conf.obs_param[14]
#        self.A3 = self.conf.obs_param[15]
#        self.B3 = self.conf.obs_param[16]
#        self.C3 = self.conf.obs_param[17]#

#        self.TARGET_STATE = self.conf.TARGET_STATE#

#        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
#        cx = casadi.SX.sym("x",self.nx,1)
#        cu = casadi.SX.sym("u",self.nu,1)#

#        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])#

#        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cx[:self.nq])
#        self.p_ee = casadi.Function('p_ee', [cx], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])#

#        if self.name == 'running_model':
#            self.weights = np.copy(self.conf.cost_weights_running)
#        elif self.name == 'terminal_model':
#            self.weights = np.copy(self.conf.cost_weights_terminal)
#        else:
#            print("The model can be either 'running_model' or 'terminal_model'")
#            import sys
#            sys.exit()#

#        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])#

#        #self.step = casadi.Function('step', [cx,cu], [self.step_fun(cx,cu)])#

#    def cost_fun(self, x, u):
#        ''' Compute cost '''
#        p_ee = self.p_ee(x)#

#        ### Penalties representing the obstacle ###
#        ell1_cost = np.log(np.exp(self.conf.alpha*-(((p_ee[0]-self.conf.XC1)**2)/((self.conf.A1/2)**2) + ((p_ee[1]-self.conf.YC1)**2)/((self.conf.B1/2)**2) + ((p_ee[2]-self.conf.ZC1)**2)/((self.conf.C1/2)**2) - 1.0)) + 1)/self.conf.alpha
#        ell2_cost = np.log(np.exp(self.conf.alpha*-(((p_ee[0]-self.conf.XC2)**2)/((self.conf.A2/2)**2) + ((p_ee[1]-self.conf.YC2)**2)/((self.conf.B2/2)**2) + ((p_ee[2]-self.conf.ZC2)**2)/((self.conf.C2/2)**2) - 1.0)) + 1)/self.conf.alpha
#        ell3_cost = np.log(np.exp(self.conf.alpha*-(((p_ee[0]-self.conf.XC3)**2)/((self.conf.A3/2)**2) + ((p_ee[1]-self.conf.YC3)**2)/((self.conf.B3/2)**2) + ((p_ee[2]-self.conf.ZC3)**2)/((self.conf.C3/2)**2) - 1.0)) + 1)/self.conf.alpha#

#        ### Control effort term ###
#        u_cost = 0
#        for i in range(self.nu):
#            u_cost += u[i]**2#

#        ### Distence to target term (log valley centered at target) ###
#        peak_rew = np.log(np.exp(self.conf.alpha2*-(np.sqrt((p_ee[0]-self.conf.TARGET_STATE[0])**2 +0.1) - np.sqrt(0.1) - 0.1 + np.sqrt((p_ee[1]-self.conf.TARGET_STATE[1])**2 +0.1) - np.sqrt(0.1) - 0.1 + np.sqrt((p_ee[2]-self.conf.TARGET_STATE[2])**2 +0.1) - np.sqrt(0.1) - 0.1)) + 1)/self.conf.alpha2#

#        ### Distence to target term (quadratic term) ###
#        dist_cost = (p_ee[0]-self.conf.TARGET_STATE[0])**2 + (p_ee[1]-self.conf.TARGET_STATE[1])**2 + (p_ee[2]-self.conf.TARGET_STATE[2])**2#

#        ### Terminal cost on final velocity ###
#        v_cost = 0
#        for i in range(self.nv):
#            v_cost += x[i+self.nq]**2
#        
#        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
# 
#        return cost
#    
#    def simulate_fun(self, x, u):
#        ''' Integrate dynamics '''
#        a = cpin.aba(self.conf.cmodel,self.conf.cdata,x[:self.nq],x[self.nq:],u)
#        
#        if self.conf.integration_scheme == 'SI-Euler':
#            F = casadi.vertcat(x[self.nq:]+self.dt*a, a)
#        elif self.conf.integration_scheme == 'E-Euler':
#            F = casadi.vertcat(x[self.nq:], a)
#        x_next = x + self.conf.dt*F
#        
#        return x_next
#    
#    def step_fun(self, x, u):
#        ''' Return next state and cost '''
#        cost = self.cost(x, u)
#        
#        x_next = self.x_next(x,u)
#        
#        return x_next, cost