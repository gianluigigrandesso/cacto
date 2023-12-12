import sys
import math
import casadi
import numpy as np
import pinocchio.casadi as cpin
    
class SingleIntegrator_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf
        
        self.nx = self.conf.nx
        self.nu = self.conf.na

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)

        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])

        self.p_ee = casadi.Function('p_ee', [cx], [self.get_end_effector_position_fun(cx)])

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])

    def get_end_effector_position_fun(self, cx):      
        p_ee = casadi.SX(3,1)
        p_ee[:2] = cx[:2]
        p_ee[-1] = 0  

        return p_ee
    
    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**10

        return u_cost
    
    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Control effort term ###
        if u is not None:
            u_cost = self.bound_control_cost(u)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun(self, x, u):
        ''' Integrate dynamics '''
        x_next = casadi.SX(self.conf.nx,1)
        x_next[0] = x[0] + self.conf.dt*u[0]
        x_next[1] = x[1] + self.conf.dt*u[1]
        
        return x_next
    
    def step_fun(self, x, u):
        ''' Return next state and cost '''
        cost = self.cost(x, u)
        
        x_next = self.x_next(x,u)
        
        return x_next, cost

class DoubleIntegrator_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf
        
        self.nq = self.conf.cmodel.nq
        self.nv = self.conf.cmodel.nv
        self.nx = self.nq+self.nv
        self.nu = self.conf.cmodel.nv

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)

        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])

        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cx[:self.nq])
        self.p_ee = casadi.Function('p_ee', [cx], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])

    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**10

        return u_cost

    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Control effort term ###
        if u is not None:
            u_cost = self.bound_control_cost(u)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
        
        ### Terminal cost on final velocity ###
        v_cost = 0
        for i in range(self.nv):
            v_cost += x[i+self.nq]**2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun(self, x, u):
        ''' Integrate dynamics '''
        a = cpin.aba(self.conf.cmodel,self.conf.cdata,x[:self.nq],x[self.nq:],u)
        
        if self.conf.integration_scheme == 'SI-Euler':
            F = casadi.vertcat(x[self.nq:]+self.dt*a, a)
        elif self.conf.integration_scheme == 'E-Euler':
            F = casadi.vertcat(x[self.nq:], a)
        x_next = x + self.conf.dt*F
        
        return x_next
    
    def step_fun(self, x, u):
        ''' Return next state and cost '''
        cost = self.cost(x, u)
        
        x_next = self.x_next(x,u)
        
        return x_next, cost
    
class Car_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf
        
        self.nx = self.conf.nx
        self.nu = self.conf.na

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)

        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])

        self.p_ee = casadi.Function('p_ee', [cx], [self.get_end_effector_position_fun(cx)])

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])

    def get_end_effector_position_fun(self, cx):        
        p_ee = casadi.SX(3,1)
        p_ee[:2] = cx[:2]
        p_ee[-1] = 0  

        return p_ee
    
    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**10

        return u_cost
    
    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Control effort term ###
        if u is not None:
            u_cost = self.bound_control_cost(u)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun(self, x, u):
        ''' Integrate dynamics '''
        x_next = casadi.SX(self.conf.nx,1)
        x_next[0] = x[0] + self.conf.dt*x[3]*casadi.cos(x[2]) + self.conf.dt**2*x[4]*casadi.cos(x[2])/2
        x_next[1] = x[1] + self.conf.dt*x[3]*casadi.sin(x[2]) + self.conf.dt**2*x[4]*casadi.sin(x[2])/2
        x_next[2] = x[2] + self.conf.dt*u[0]
        x_next[3] = x[3] + self.conf.dt*x[4]
        x_next[4] = x[4] + self.conf.dt*u[1]
        
        return x_next
    
    def step_fun(self, x, u):
        ''' Return next state and cost '''
        cost = self.cost(x, u)
        
        x_next = self.x_next(x,u)
        
        return x_next, cost
    
class CarPark_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf
        
        self.nx = self.conf.nx
        self.nu = self.conf.na

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)

        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])

        self.p_ee = casadi.Function('p_ee', [cx], [self.get_end_effector_position_fun(cx)])

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])

    def get_end_effector_position_fun(self, cx):       
        p_ee = casadi.SX(3,1) 
        p_ee[:2] = cx[:2] + np.array([[np.cos(cx[2]), -np.sin(cx[2])], [np.sin(cx[2]), np.cos(cx[2])]]).dot(np.array([self.conf.L_delta/2,0]))
        p_ee[-1] = 0  

        return p_ee
    
    def obs_cost_fun(self, x, y, x_clip, y_clip, Wx, Wy, fv=1, k=50):
        k = self.conf.k_db
        obs_cost = (4 + 4 * (y - y_clip + Wy / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * fv * (-np.sqrt(4 + 4 * (y - y_clip - Wy / 2) ** 2 * k ** 2) / 2 + (y - y_clip - Wy / 2) * k) * (4 + 4 * (x - x_clip + Wx / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * (4 + 4 * (y - y_clip - Wy / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * (np.sqrt(4 + 4 * (y - y_clip + Wy / 2) ** 2 * k ** 2) / 2 + (y - y_clip + Wy / 2) * k) * (4 + 4 * (x - x_clip - Wx / 2) ** 2 * k ** 2) ** (-0.1e1 / 0.2e1) * (np.sqrt(4 + 4 * (x - x_clip + Wx / 2) ** 2 * k ** 2) / 2 + (x - x_clip + Wx / 2) * k) * (-np.sqrt(4 + 4 * (x - x_clip - Wx / 2) ** 2 * k ** 2) / 2 + (x - x_clip - Wx / 2) * k)

        return obs_cost
    
    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**10

        return u_cost
    
    def rotation_matrix(self, angle):
        ''' Compute the 2x2 rotation matrix for a given angle '''
        cos_theta = casadi.cos(angle)
        sin_theta = casadi.sin(angle)

        rotation_mat = casadi.vertcat(casadi.horzcat(cos_theta, -sin_theta),
                                   casadi.horzcat(sin_theta, cos_theta))
        return rotation_mat

    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee_tmp = self.p_ee(x)
        p_ee = p_ee_tmp[:2]
        theta_ee = x[2]

        ### Penalties representing the obstacle ###
        obs_cost = 0
        check_points_WF = casadi.mtimes(self.rotation_matrix(theta_ee), self.conf.check_points_BF.T).T + casadi.repmat(p_ee.T, self.conf.check_points_BF.shape[0], 1)
        obs_cost += casadi.sum1(self.obs_cost_fun(check_points_WF[:, 0], check_points_WF[:, 1], self.XC1, self.YC1, self.A1, self.B1))
        obs_cost += casadi.sum1(self.obs_cost_fun(check_points_WF[:, 0], check_points_WF[:, 1], self.XC2, self.YC2, self.A2, self.B2))
        obs_cost += casadi.sum1(self.obs_cost_fun(check_points_WF[:, 0], check_points_WF[:, 1], self.XC3, self.YC3, self.A3, self.B3))
             
        ### Control effort term ###
        u_cost = self.bound_control_cost(u)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 2*0.1 - 2*np.sqrt(0.1))) + 1)/self.alpha2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*x[3]**2 + self.weights[3]*obs_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun(self, x, u):
        ''' Integrate dynamics '''
        x_next = casadi.SX(self.conf.nx,1)
        x_next[0] = x[0] + self.conf.dt*x[3]*casadi.cos(x[2])
        x_next[1] = x[1] + self.conf.dt*x[3]*casadi.sin(x[2])
        x_next[2] = x[2] + self.conf.dt*x[3]*casadi.tan(x[4])/self.conf.L_delta
        x_next[3] = x[3] + self.conf.dt*u[0]
        x_next[4] = x[4] + self.conf.dt*u[1]/self.conf.tau_delta
        
        return x_next
        
        return x_next
    
    def step_fun(self, x, u):
        ''' Return next state and cost '''
        cost = self.cost(x, u)
        
        x_next = self.x_next(x,u)
        
        return x_next, cost

class Manipulator_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf
        
        self.nq = self.conf.cmodel.nq
        self.nv = self.conf.cmodel.nv
        self.nx = self.nq+self.nv
        self.nu = self.conf.cmodel.nv

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)

        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])

        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cx[:self.nq])
        self.p_ee = casadi.Function('p_ee', [cx], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])

    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**10

        return u_cost
    
    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Control effort term ###
        if u is not None:
            u_cost = self.bound_control_cost(u)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
        
        ### Terminal cost on final velocity ###
        v_cost = 0
        for i in range(self.nv):
            v_cost += x[i+self.nq]**2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun(self, x, u):
        ''' Integrate dynamics '''
        a = cpin.aba(self.conf.cmodel,self.conf.cdata,x[:self.nq],x[self.nq:],u)
        
        if self.conf.integration_scheme == 'SI-Euler':
            F = casadi.vertcat(x[self.nq:]+self.dt*a, a)
        elif self.conf.integration_scheme == 'E-Euler':
            F = casadi.vertcat(x[self.nq:], a)
        x_next = x + self.conf.dt*F
        
        return x_next
    
    def step_fun(self, x, u):
        ''' Return next state and cost '''
        cost = self.cost(x, u)
        
        x_next = self.x_next(x,u)
        
        return x_next, cost

class UR5_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf
        
        self.nq = self.conf.cmodel.nq
        self.nv = self.conf.cmodel.nv
        self.nx = self.nq+self.nv
        self.nu = self.conf.cmodel.nv

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.ZC1 = self.conf.obs_param[2]
        self.XC2 = self.conf.obs_param[3]
        self.YC2 = self.conf.obs_param[4]
        self.ZC2 = self.conf.obs_param[5]
        self.XC3 = self.conf.obs_param[6]
        self.YC3 = self.conf.obs_param[7]
        self.ZC3 = self.conf.obs_param[8]
        
        self.A1 = self.conf.obs_param[9]
        self.B1 = self.conf.obs_param[10]
        self.C1 = self.conf.obs_param[11]
        self.A2 = self.conf.obs_param[12]
        self.B2 = self.conf.obs_param[13]
        self.C2 = self.conf.obs_param[14]
        self.A3 = self.conf.obs_param[15]
        self.B3 = self.conf.obs_param[16]
        self.C3 = self.conf.obs_param[17]

        self.TARGET_STATE = self.conf.TARGET_STATE

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)

        self.x_next = casadi.Function('x_next', [cx, cu], [self.simulate_fun(cx,cu)])

        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cx[:self.nq])
        self.p_ee = casadi.Function('p_ee', [cx], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.cost = casadi.Function('cost', [cx,cu], [self.cost_fun(cx,cu)])

    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = np.log(np.exp(self.conf.alpha*-(((p_ee[0]-self.conf.XC1)**2)/((self.conf.A1/2)**2) + ((p_ee[1]-self.conf.YC1)**2)/((self.conf.B1/2)**2) + ((p_ee[2]-self.conf.ZC1)**2)/((self.conf.C1/2)**2) - 1.0)) + 1)/self.conf.alpha
        ell2_cost = np.log(np.exp(self.conf.alpha*-(((p_ee[0]-self.conf.XC2)**2)/((self.conf.A2/2)**2) + ((p_ee[1]-self.conf.YC2)**2)/((self.conf.B2/2)**2) + ((p_ee[2]-self.conf.ZC2)**2)/((self.conf.C2/2)**2) - 1.0)) + 1)/self.conf.alpha
        ell3_cost = np.log(np.exp(self.conf.alpha*-(((p_ee[0]-self.conf.XC3)**2)/((self.conf.A3/2)**2) + ((p_ee[1]-self.conf.YC3)**2)/((self.conf.B3/2)**2) + ((p_ee[2]-self.conf.ZC3)**2)/((self.conf.C3/2)**2) - 1.0)) + 1)/self.conf.alpha

        ### Control effort term ###
        u_cost = 0
        for i in range(self.nu):
            u_cost += u[i]**2 + self.conf.w_b*(u[i]/self.conf.u_max[i])**10

        ### Distence to target term (log valley centered at target) ###
        peak_rew = np.log(np.exp(self.conf.alpha2*-(np.sqrt((p_ee[0]-self.conf.TARGET_STATE[0])**2 +0.1) - np.sqrt(0.1) - 0.1 + np.sqrt((p_ee[1]-self.conf.TARGET_STATE[1])**2 +0.1) - np.sqrt(0.1) - 0.1 + np.sqrt((p_ee[2]-self.conf.TARGET_STATE[2])**2 +0.1) - np.sqrt(0.1) - 0.1)) + 1)/self.conf.alpha2

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.conf.TARGET_STATE[0])**2 + (p_ee[1]-self.conf.TARGET_STATE[1])**2 + (p_ee[2]-self.conf.TARGET_STATE[2])**2

        ### Terminal cost on final velocity ###
        v_cost = 0
        for i in range(self.nv):
            v_cost += x[i+self.nq]**2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun(self, x, u):
        ''' Integrate dynamics '''
        a = cpin.aba(self.conf.cmodel,self.conf.cdata,x[:self.nq],x[self.nq:],u)
        
        if self.conf.integration_scheme == 'SI-Euler':
            F = casadi.vertcat(x[self.nq:]+self.dt*a, a)
        elif self.conf.integration_scheme == 'E-Euler':
            F = casadi.vertcat(x[self.nq:], a)
        x_next = x + self.conf.dt*F
        
        return x_next
    
    def step_fun(self, x, u):
        ''' Return next state and cost '''
        cost = self.cost(x, u)
        
        x_next = self.x_next(x,u)
        
        return x_next, cost