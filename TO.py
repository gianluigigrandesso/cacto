import sys
import math
import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from pyomo.dae import *
from pyomo.environ import *
from inits import init_action, init_state, init_state_ICS, init_0

class TO_Pyomo:

    def __init__(self, env, conf, system_id, w_S=0):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param u_min :                      (float array) Action lower bound array
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param dt :                         (float) Timestep

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array

            :param EPISODE_ICS_INIT :           (int) Episodes where ICS warm-starting is used instead of actor rollout

        :input system_id :                      (str) Id system

        :input w_S :                            (float) Sobolev-training weight
        '''

        self.env = env
        self.conf = conf

        self.system_id = system_id

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

        self.w_S = w_S

        if self.w_S != 0:
            self.CAMS = CasadiActionModelSystem

        return

    def TO_Manipulator_Model(self, ICS, init_x, init_a, init_tau, T, init_TO=None):
        ''' Create TO pyomo model - manipulator '''
        m = ConcreteModel()
        m.k = RangeSet(0, T)
        m.ku = RangeSet(0, T-1)

        if init_TO != None:
            init_TO_controls = init_TO[0]
            init_TO_states = init_TO[1]      
            m.tau0 = Var(m.ku, initialize=init_tau(m,m.ku,0,init_TO_controls), bounds=(self.conf.u_min[0], self.conf.u_max[0])) 
            m.tau1 = Var(m.ku, initialize=init_tau(m,m.ku,1,init_TO_controls), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.tau2 = Var(m.ku, initialize=init_tau(m,m.ku,2,init_TO_controls), bounds=(self.conf.u_min[2], self.conf.u_max[2])) 
            m.q0 = Var(m.k, initialize=init_x(m,m.k,0,init_TO_states))
            m.q1 = Var(m.k, initialize=init_x(m,m.k,1,init_TO_states))
            m.q2 = Var(m.k, initialize=init_x(m,m.k,2,init_TO_states))
            m.v0 = Var(m.k, initialize=init_x(m,m.k,3,init_TO_states))
            m.v1 = Var(m.k, initialize=init_x(m,m.k,4,init_TO_states))
            m.v2 = Var(m.k, initialize=init_x(m,m.k,5,init_TO_states))
        else:    
            m.tau0 = Var(m.ku, initialize=init_tau(m,m.ku,0), bounds=(self.conf.u_min[0], self.conf.u_max[0]))
            m.tau1 = Var(m.ku, initialize=init_tau(m,m.ku,1), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.tau2 = Var(m.ku, initialize=init_tau(m,m.ku,2), bounds=(self.conf.u_min[2], self.conf.u_max[2]))
            m.q0 = Var(m.k, initialize=init_x(m,m.k,0,ICS))
            m.q1 = Var(m.k, initialize=init_x(m,m.k,1,ICS))
            m.q2 = Var(m.k, initialize=init_x(m,m.k,2,ICS))
            m.v0 = Var(m.k, initialize=init_x(m,m.k,3,ICS))
            m.v1 = Var(m.k, initialize=init_x(m,m.k,4,ICS))
            m.v2 = Var(m.k, initialize=init_x(m,m.k,5,ICS))
        
        m.a0 = Var(m.k, initialize=init_a)
        m.a1 = Var(m.k, initialize=init_a)
        m.a2 = Var(m.k, initialize=init_a)

        m.icfix_q0 = Constraint(rule = lambda m: m.q0[0] == ICS[0])
        m.icfix_q1 = Constraint(rule = lambda m: m.q1[0] == ICS[1])
        m.icfix_q2 = Constraint(rule = lambda m: m.q2[0] == ICS[2])
        m.icfix_v0 = Constraint(rule = lambda m: m.v0[0] == ICS[3])        
        m.icfix_v1 = Constraint(rule = lambda m: m.v1[0] == ICS[4])        
        m.icfix_v2 = Constraint(rule = lambda m: m.v2[0] == ICS[5])        

        ### Integration ###
        m.v0_update = Constraint(m.k, rule = lambda m, k:
           m.v0[k+1] == m.v0[k] + self.conf.dt*m.a0[k] if k < T else Constraint.Skip)

        m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + self.conf.dt*m.a1[k] if k < T else Constraint.Skip)

        m.v2_update = Constraint(m.k, rule = lambda m, k:
           m.v2[k+1] == m.v2[k] + self.conf.dt*m.a2[k] if k < T else Constraint.Skip)

        m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + self.conf.dt*m.v0[k] if k < T else Constraint.Skip)

        m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + self.conf.dt*m.v1[k] if k < T else Constraint.Skip)

        m.q2_update = Constraint(m.k, rule = lambda m, k:
           m.q2[k+1] == m.q2[k] + self.conf.dt*m.v2[k] if k < T else Constraint.Skip)


        ### Dyamics ###
        m.EoM_0 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*self.conf.Iz*m.a2[k] + self.conf.l**2*self.conf.M*m.a2[k] + 2*self.conf.l**2*self.conf.M*m.a2[k]*cos(m.q2[k]) + 2*self.conf.l**2*self.conf.M*m.a2[k]*cos(m.q1[k] + m.q2[k]) + 2*m.a1[k]*(4*self.conf.Iz + 3*self.conf.l**2*self.conf.M + 3*self.conf.l**2*self.conf.M*cos(m.q1[k]) + 2*self.conf.l**2*self.conf.M*cos(m.q2[k]) + self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k])) + m.a0[k]*(12*self.conf.Iz + 15*self.conf.l**2*self.conf.M +
    12*self.conf.l**2*self.conf.M*cos(m.q1[k]) + 4*self.conf.l**2*self.conf.M*cos(m.q2[k]) + 4*self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k])) - 4*m.tau0[k] - 12*self.conf.l**2*self.conf.M*sin(m.q1[k])*m.v0[k]*m.v1[k] - 4*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]*m.v1[k] - 6*self.conf.l**2*self.conf.M*sin(m.q1[k])*m.v1[k]**2 - 2*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v1[k]**2
    - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v1[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v1[k]*m.v2[k] - 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v2[k]**2 - 2*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v2[k]**2) == 0 if k < T else Constraint.Skip)

        m.EoM_1 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*self.conf.Iz*m.a2[k] + self.conf.l**2*self.conf.M*m.a2[k] + 2*self.conf.l**2*self.conf.M*m.a2[k]*cos(m.q2[k]) + m.a1[k]*(8*self.conf.Iz + 6*self.conf.l**2*self.conf.M + 4*self.conf.l**2*self.conf.M*cos(m.q2[k])) + 2*m.a0[k]*(4*self.conf.Iz + 3*self.conf.l**2*self.conf.M + 3*self.conf.l**2*self.conf.M*cos(m.q1[k]) + 2*self.conf.l**2*self.conf.M*cos(m.q2[k]) + self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k]))
    - 4*m.tau1[k] + 6*self.conf.l**2*self.conf.M*sin(m.q1[k])*m.v0[k]**2 + 2*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]**2 - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v1[k]*m.v2[k] - 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v2[k]**2) == 0 if k < T else Constraint.Skip)

        m.EoM_2 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*self.conf.Iz*m.a2[k] + self.conf.l**2*self.conf.M*m.a2[k] + m.a1[k]*(4*self.conf.Iz + self.conf.l**2*self.conf.M + 2*self.conf.l**2*self.conf.M*cos(m.q2[k])) + m.a0[k]*(4*self.conf.Iz + self.conf.l**2*self.conf.M + 2*self.conf.l**2*self.conf.M*cos(m.q2[k]) + 2*self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k])) - 4*m.tau2[k] + 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]**2 + 2*self.conf.l**2*self.conf.M*sin(m.q1[k] +
    m.q2[k])*m.v0[k]**2 + 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]*m.v1[k] + 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v1[k]**2) == 0 if k < T else Constraint.Skip)

        ### Penalties representing the obstacle ###
        m.ell1_cost = sum((log(exp(self.alpha*-((((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-self.XC1)**2)/((self.A1/2)**2) + (((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  for k in m.k)
        m.ell2_cost = sum((log(exp(self.alpha*-((((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-self.XC2)**2)/((self.A2/2)**2) + (((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha)  for k in m.k)
        m.ell3_cost = sum((log(exp(self.alpha*-((((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-self.XC3)**2)/((self.A3/2)**2) + (((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)  for k in m.k)

        ### Control effort term ###
        m.u_cost = sum((m.tau0[k]**2 + m.tau1[k]**2 + m.tau2[k]**2) for k in m.ku)
    
        ### Distence to target term (quadratic term) ###
        m.dist_cost = sum(((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-self.x_des)**2 + ((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-self.y_des)**2 for k in m.k) 
        
        ### Distence to target term (log valley centered at target)###
        m.peak_rew = sum((log(exp(self.alpha2*-(sqrt(((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-self.x_des)**2 +0.1) - 0.1 + sqrt(((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-self.y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/self.alpha2) for k in m.k) 

        ### Final velocity term ###
        m.v_cost = (m.v0[T])**2 + (m.v1[T])**2 + (m.v2[T])**2

        m.obj = Objective(expr = self.scale*(self.conf.w_d*m.dist_cost - self.conf.w_peak*m.peak_rew + self.conf.w_v*m.v_cost + self.conf.w_ob1*m.ell1_cost + self.conf.w_ob2*m.ell2_cost + self.conf.w_ob3*m.ell3_cost + self.conf.w_u*m.u_cost - sum(self.offset for _ in m.k)), sense=minimize)
        return m

    def TO_Manipulator_Solve(self, ep, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Solve TO problem - manipulator '''
        if ep < self.conf.EPISODE_ICS_INIT:
            TO_mdl = self.TO_Manipulator_Model(ICS_state, init_state_ICS, init_0, init_0, T)
        else:
            TO_mdl = self.TO_Manipulator_Model(ICS_state, init_state, init_0, init_action, T, init_TO = [init_TO_controls, init_TO_states])

        # Indexes of TO variables      
        K =  np.array([k for k in TO_mdl.k])
        Ku = np.array([k for k in TO_mdl.ku]) 

        # Set solver options
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = "ma57"
        
        ### Try to solve TO problem
        try:
            results = solver.solve(TO_mdl,tee=False)                                
            if str(results.solver.termination_condition) == "optimal":    
                #Retrieve control trajectory
                TO_controls = np.empty((len(Ku), self.conf.nb_action))
                TO_controls[:,0] = [TO_mdl.tau0[k]() for k in Ku]
                TO_controls[:,1] = [TO_mdl.tau1[k]() for k in Ku]
                TO_controls[:,2] = [TO_mdl.tau2[k]() for k in Ku]

                TO_states = np.empty((len(K), self.conf.nb_state))
                TO_states[:,0] = [TO_mdl.q0[k]() for k in K ]
                TO_states[:,1] = [TO_mdl.q1[k]() for k in K ]
                TO_states[:,2] = [TO_mdl.q2[k]() for k in K ]
                TO_states[:,3] = [TO_mdl.v0[k]() for k in K ]
                TO_states[:,4] = [TO_mdl.v1[k]() for k in K ]
                TO_states[:,5] = [TO_mdl.v2[k]() for k in K ]

                success_flag = 1
            else:
                print('TO solution not optimal')     
                raise Exception()         
        except:
            success_flag = 0
            TO_controls = None
            TO_states = None
            print("*** TO failed ***")  

        ee_pos_arr = np.empty((len(K),3))
        for k in range(len(K)):
            ee_pos_arr[k,:] = self.env.get_end_effector_position(np.array([TO_mdl.q0[k](), TO_mdl.q1[k](), TO_mdl.q2[k]()]))

        return success_flag, TO_controls, TO_states, ee_pos_arr

    def TO_DoubleIntegrator_Model(self,ICS, init_x, init_a, init_tau, T, init_TO=None):
        ''' Create TO pyomo model - double integrator '''
        m = ConcreteModel()
        m.k = RangeSet(0, T)
        m.ku = RangeSet(0, T-1)
        
        if init_TO != None:
            init_TO_controls = init_TO[0]
            init_TO_states = init_TO[1]     
            m.tau0 = Var(m.ku, initialize=init_tau(m,m.ku,0,init_TO_controls), bounds=(self.conf.u_min[0], self.conf.u_max[0])) 
            m.tau1 = Var(m.ku, initialize=init_tau(m,m.ku,1,init_TO_controls), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.q0 = Var(m.k, initialize=init_x(m,m.k,0,init_TO_states))
            m.q1 = Var(m.k, initialize=init_x(m,m.k,1,init_TO_states))
            m.v0 = Var(m.k, initialize=init_x(m,m.k,2,init_TO_states))
            m.v1 = Var(m.k, initialize=init_x(m,m.k,3,init_TO_states))
        else:   
            m.tau0 = Var(m.k, initialize=init_tau(m,m.k,0), bounds=(self.conf.u_min[0], self.conf.u_max[0])) 
            m.tau1 = Var(m.k, initialize=init_tau(m,m.k,1), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.q0 = Var(m.k, initialize=init_x(m,m.k,0,ICS))
            m.q1 = Var(m.k, initialize=init_x(m,m.k,1,ICS))
            m.v0 = Var(m.k, initialize=init_x(m,m.k,2,ICS))
            m.v1 = Var(m.k, initialize=init_x(m,m.k,3,ICS))

        m.icfix_q0 = Constraint(rule = lambda m: m.q0[0] == ICS[0])
        m.icfix_q1 = Constraint(rule = lambda m: m.q1[0] == ICS[1])
        m.icfix_v0 = Constraint(rule = lambda m: m.v0[0] == ICS[2])
        m.icfix_v1 = Constraint(rule = lambda m: m.v1[0] == ICS[3])        

        ### Integration ###
        m.v0_update = Constraint(m.k, rule = lambda m, k:
           m.v0[k+1] == m.v0[k] + self.conf.dt*m.tau0[k] if k < T else Constraint.Skip) #

        m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + self.conf.dt*m.tau1[k] if k < T else Constraint.Skip) #

        m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + self.conf.dt*m.v0[k] if k < T else Constraint.Skip) # + self.conf.dt**2*m.tau0[k]/2

        m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + self.conf.dt*m.v1[k]  if k < T else Constraint.Skip) #+ self.conf.dt**2*m.tau1[k]/2 

        ### Penalties representing the obstacle ###
        m.ell1_cost = sum((log(exp(self.alpha*-(((m.q0[k]-self.XC1)**2)/((self.A1/2)**2) + ((m.q1[k]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  for k in m.k)
        m.ell2_cost = sum((log(exp(self.alpha*-(((m.q0[k]-self.XC2)**2)/((self.A2/2)**2) + ((m.q1[k]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha)  for k in m.k)
        m.ell3_cost = sum((log(exp(self.alpha*-(((m.q0[k]-self.XC3)**2)/((self.A3/2)**2) + ((m.q1[k]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)  for k in m.k)

        ### Control effort term ###
        m.u_cost = sum((m.tau0[k]**2 + m.tau1[k]**2) for k in m.ku)

        ### Distence to target term (quadratic term) ###
        m.dist_cost = sum((m.q0[k]-self.x_des)**2 + (m.q1[k]-self.y_des)**2 for k in m.k)
        
        ### Distence to target term (log valley centered at target)###
        m.peak_rew = sum((log(exp(self.alpha2*-(sqrt((m.q0[k]-self.x_des)**2 + 0.1) - sqrt(0.1) - 0.1 + sqrt((m.q1[k]-self.y_des)**2 +0.1) - sqrt(0.1) - 0.1)) + 1)/self.alpha2) for k in m.k)
        
        m.obj = Objective(expr = self.scale*(- self.w_peak*m.peak_rew + self.w_d*m.dist_cost + self.w_ob1*m.ell1_cost + self.w_ob2*m.ell2_cost + self.w_ob3*m.ell3_cost + self.w_u*m.u_cost - sum(self.offset for _ in m.k)), sense=minimize) 

        return m

    def TO_DoubleIntegrator_Solve(self, ep, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Solve TO problem - double integrator '''               
        if ep < self.conf.EPISODE_ICS_INIT:
            TO_mdl = self.TO_DoubleIntegrator_Model(ICS_state, init_state_ICS, init_0, init_0, T)
        else:
            TO_mdl = self.TO_DoubleIntegrator_Model(ICS_state, init_state, init_0, init_action, T, init_TO = [init_TO_controls, init_TO_states])
        
        # Indexes of TO variables       
        K = np.array([k for k in TO_mdl.k]) 
        Ku = np.array([k for k in TO_mdl.ku]) 
        
        # Set solver options
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = "ma57"
        
        ### Try to solve TO problem
        try:
            results = solver.solve(TO_mdl, tee=False)                              
            if str(results.solver.termination_condition) == "optimal":    

                #Retrieve control trajectory
                TO_controls = np.empty((len(Ku), self.conf.nb_action))
                TO_controls[:,0] = [TO_mdl.tau0[k]() for k in Ku]
                TO_controls[:,1] = [TO_mdl.tau1[k]() for k in Ku]

                TO_states = np.empty((len(K), self.conf.nb_state))
                TO_states[:,0] = [TO_mdl.q0[k]() for k in K ]
                TO_states[:,1] = [TO_mdl.q1[k]() for k in K ]
                TO_states[:,2] = [TO_mdl.v0[k]() for k in K ]
                TO_states[:,3] = [TO_mdl.v1[k]() for k in K ]
                
                success_flag = 1
            else:
                print('TO solution not optimal')    
                raise Exception()         
        except:
            success_flag = 0
            TO_controls = None
            TO_states = None
            print("*** TO failed ***")  

        ee_pos_arr = np.empty((len(K),3))
        for k in range(len(K)):
            ee_pos_arr[k,:] = self.env.get_end_effector_position(np.array([TO_mdl.q0[k](), TO_mdl.q1[k]()]))
        
        return success_flag, TO_controls, TO_states, ee_pos_arr

    def TO_Solve(self, ep, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Create TO problem '''
        if self.system_id == 'double_integrator':
            success_flag, TO_controls, TO_states, ee_pos_arr = self.TO_DoubleIntegrator_Solve(ep, ICS_state, init_TO_states, init_TO_controls, T) 
        elif self.system_id == 'manipulator':
            success_flag, TO_controls, TO_states, ee_pos_arr = self.TO_Manipulator_Solve(ep, ICS_state, init_TO_states, init_TO_controls, T)
        else:
            print('Pyomo {} model not found'.format(self.system_id))

        if self.w_S != 0 & success_flag:
            self.runningSingleModel = self.CAMS('running_model', self.conf)
            self.terminalModel = self.CAMS('terminal_model', self.conf)
            dVdx = self.backward_pass(T+1, TO_states, TO_controls) 
        else:
            dVdx = np.zeros((T+1, self.conf.nb_state-1))
        
        # Plot TO solution    
        # plot_results_TO(TO_mdl)  

        return TO_controls, TO_states, success_flag, ee_pos_arr, dVdx 

    def f_d(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        q = x[:self.nq]
        v = x[self.nq:]
                
        # first compute Jacobians for continuous time dynamics
        self.Fx = np.zeros((self.nx,self.nx))
        self.Fu = np.zeros((self.nx,self.nu))

        pin.computeABADerivatives(self.model, self.data, q, v, u)

        self.Fx[:self.nv, :self.nv] = 0.0
        self.Fx[:self.nv, self.nv:] = np.identity(self.nv)
        self.Fx[self.nv:, :self.nv] = self.data.ddq_dq
        self.Fx[self.nv:, self.nv:] = self.data.ddq_dv
        self.Fu[self.nv:, :] = self.data.Minv
        
        # Convert them to discrete time
        self.Fx = np.identity(self.nx) + self.conf.dt * self.Fx
        self.Fu *= self.conf.dt
        
        return self.Fx, self.Fu
        
    def backward_pass(self, T, TO_states, TO_controls, mu=1e-4): 
        ''' Perform the backward-pass of DDP to obtain the derivatives of the Value function w.r.t x '''
        X_bar = np.zeros((T, self.nx))
        for i in range(self.nx):
            X_bar[:,i] = [TO_states[n,i] for n in range(T)]

        U_bar = np.zeros((T-1, self.nu))
        for i in range(self.nu):
            U_bar[:,i] = [TO_controls[n,i] for n in range(T-1)]

        n = self.nx
        m = self.nu
 
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
        
        x = casadi.SX.sym('x',self.conf.nb_state-1,1)
        u = casadi.SX.sym('u',self.conf.nb_action,1)

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
        V_x  = np.zeros((T, n))

        # Dynamics derivatives w.r.t. x and u
        A = np.zeros((T-1, n, n))
        B = np.zeros((T-1, n, m))
        
        # Initialize value function
        l_x[-1,:], l_xx[-1,:,:] = np.reshape(fun_terminal_cost_x(X_bar[-1,:]),self.nx), fun_terminal_cost_xx(X_bar[-1,:])
        V_xx[T-1,:,:] = l_xx[-1,:,:]
        V_x[T-1,:]    = l_x[-1,:]

        for i in range(T-2, -1, -1):
            # Compute dynamics Jacobians
            A[i,:,:], B[i,:,:] = self.f_d(X_bar[i,:], U_bar[i,:])

            # Compute the gradient of the cost function at X=X_bar
            l_x[i,:], l_xx[i,:,:] = np.reshape(fun_running_cost_x(X_bar[i,:]),self.nx), fun_running_cost_xx(X_bar[i,:])
            l_u[i,:],l_uu[i,:,:]  = np.reshape(fun_running_cost_u(U_bar[i,:]),self.nu), fun_running_cost_uu(U_bar[i,:]) 
            l_xu[i,:,:] = fun_running_cost_xu(X_bar[i,:], U_bar[i,:])                                                                
            
            # Compute regularized cost-to-go
            Q_x[i,:]     = l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            Q_u[i,:]     = l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            Q_xx[i,:,:]  = l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            Q_uu[i,:,:]  = l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            Q_xu[i,:,:]  = l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
                
            Qbar_uu       = Q_uu[i,:,:] + mu*np.identity(m)
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu)

            # Compute the derivative of the Value function w.r.t. x                
            V_x[i,:]    = Q_x[i,:]  - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_u[i,:]
            V_xx[i,:]   = Q_xx[i,:] - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_xu[i,:,:].T
        return V_x


class TO_Casadi:
    
    def __init__(self, env, conf, system_id, w_S=0):
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

        self.system_id = system_id

        self.nq = self.conf.robot.nq
        self.nv = self.conf.robot.nv
        self.nx = self.nq + self.nv
        self.nu = self.conf.robot.na
        self.model = self.conf.robot.model
        self.data = self.conf.robot.data

        self.w_S = w_S

        self.CAMS = CasadiActionModelSystem
    
    def TO_System_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Create and solbe TO casadi problem '''
        ### PROBLEM
        opti = casadi.Opti()

        # The control models are stored as a collection of shooting nodes called running models, with an additional terminal model.
        self.runningSingleModel = self.CAMS('running_model', self.conf)
        runningModels = [ self.runningSingleModel for t in range(T) ]
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
            for i in range(self.conf.nb_action):
                opti.subject_to(opti.bounded(self.conf.u_min[i], us[t][i], self.conf.u_max[i])) # control is limited
        r_cost_final = self.terminalModel.cost(xs[-1], us[-1])
        total_cost += r_cost_final
        
        ### SOLVE
        opti.minimize(total_cost)  
        
        # Create warmstart
        init_x_TO = [np.array(init_TO_states[:-1,i]) for i in range(T+1)]
        init_u_TO = [np.array(init_TO_controls[:,i]) for i in range(T)]
        
        for x,xg in zip(xs,init_x_TO): opti.set_initial(x,xg)
        for u,ug in zip(us,init_u_TO): opti.set_initial(u,ug)

        # Set solver options
        opts = {'ipopt.linear_solver':'ma57', 'ipopt.sb': 'yes','ipopt.print_level': 0, 'print_time': 0} 
        opti.solver("ipopt", opts) 
        
        try:
            opti.solve()
            TO_states = np.array([ opti.value(x) for x in xs ])
            TO_controls = np.array([ opti.value(u) for u in us ])
            TO_total_cost = opti.value(total_cost)
            ee_pos_arr = np.empty((T+1,3))
            for n in range(T+1):
                ee_pos_arr[n,:] = self.env.get_end_effector_position(TO_states[n,:])
            success_flag = 1
        except:
            print('ERROR in convergence, returning debug values')
            TO_states = np.array([ opti.debug.value(x) for x in xs ])
            TO_controls = np.array([ opti.debug.value(u) for u in us ])
            TO_total_cost = None
            ee_pos_arr = None
            success_flag = 0

        return success_flag, TO_controls, TO_states, ee_pos_arr, TO_total_cost
    
    def TO_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Retrieve TO problem solution and compute the value function derviative with respect to the state '''
        success_flag, TO_controls, TO_states, ee_pos_arr, _ = self.TO_System_Solve(ICS_state, init_TO_states, init_TO_controls, T)
        
        # Plot TO solution    
        # plot_results_TO(TO_mdl)  

        TO_states = np.concatenate((TO_states, init_TO_states[-1,0] + np.transpose(self.conf.dt*np.array([range(T+1)]))), axis=1)
        TO_states_augmented = np.concatenate((TO_states, np.ones((T+1,1))), axis=1)
        if self.w_S != 0:
            dVdx = self.backward_pass(T+1, TO_states_augmented, TO_controls) 
        else:
            dVdx = np.zeros((T+1, self.conf.nb_state))
        return TO_controls, TO_states, success_flag, ee_pos_arr, dVdx 
    
    def f_d(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        q = x[:self.nq]
        v = x[self.nq:self.nx]
                
        # first compute Jacobians for continuous time dynamics
        self.Fx = np.zeros((self.conf.nb_state+1,self.conf.nb_state+1))
        self.Fu = np.zeros((self.conf.nb_state+1,self.conf.nb_action))

        pin.computeABADerivatives(self.model, self.data, q, v, u)

        self.Fx[:self.nv, :self.nv] = 0.0
        self.Fx[:self.nv, self.nv:self.nx] = np.identity(self.nv)
        self.Fx[self.nv:self.nx, :self.nv] = self.data.ddq_dq
        self.Fx[self.nv:self.nx, self.nv:self.nx] = self.data.ddq_dv
        self.Fx[-2,-1] = 1.0
        self.Fx[-1,-1] = 0.0
        self.Fu[self.nv:self.nx, :] = self.data.Minv
        
        # Convert them to discrete time
        self.Fx = np.identity(self.conf.nb_state+1) + self.conf.dt * self.Fx
        self.Fu *= self.conf.dt
        
        return self.Fx, self.Fu
        
    def backward_pass(self, T, TO_states, TO_controls, mu=1e-4):
        ''' Perform the backward-pass of DDP to obtain the derivatives of the Value function w.r.t x '''
        n = self.conf.nb_state+1
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

        running_cost = -self.runningSingleModel.cost_BP(x, u)
        terminal_cost = -self.terminalModel.cost_BP(x, u)

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
        V_x  = np.zeros((T, n))

        # Dynamics derivatives w.r.t. x and u
        A = np.zeros((T-1, n, n))
        B = np.zeros((T-1, n, m))
        
        # Initialize value function
        l_x[-1,:], l_xx[-1,:,:] = np.reshape(fun_terminal_cost_x(X_bar[-1,:]),n), fun_terminal_cost_xx(X_bar[-1,:])
        V_xx[T-1,:,:] = l_xx[-1,:,:]
        V_x[T-1,:]    = l_x[-1,:]

        for i in range(T-2, -1, -1):
            # Compute dynamics Jacobians
            A[i,:,:], B[i,:,:] = self.f_d(X_bar[i,:], U_bar[i,:])

            # Compute the gradient of the cost function at X=X_bar
            l_x[i,:], l_xx[i,:,:] = np.reshape(fun_running_cost_x(X_bar[i,:]),n), fun_running_cost_xx(X_bar[i,:])
            l_u[i,:],l_uu[i,:,:]  = np.reshape(fun_running_cost_u(U_bar[i,:]),m), fun_running_cost_uu(U_bar[i,:]) 
            l_xu[i,:,:] = fun_running_cost_xu(X_bar[i,:], U_bar[i,:])                                                                
            
            # Compute regularized cost-to-go
            Q_x[i,:]     = l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            Q_u[i,:]     = l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            Q_xx[i,:,:]  = l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            Q_uu[i,:,:]  = l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            Q_xu[i,:,:]  = l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
                
            Qbar_uu       = Q_uu[i,:,:] + mu*np.identity(m)
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu)

            # Compute the derivative of the Value function w.r.t. x                
            V_x[i,:]    = Q_x[i,:]  - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_u[i,:]
            V_xx[i,:]   = Q_xx[i,:] - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_xu[i,:,:].T
        return V_x[:,:self.conf.nb_state]
    
class CasadiActionModelSystem:
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
        cxt = casadi.SX.sym("xt",self.nx+2,1)
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

        self.cost_BP = casadi.Function('cost', [cxt,cu], [self.cost_fun_BP(cxt,cu)])

    def cost_fun(self, x, u):
        ''' Compute cost '''
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Control effort term ###
        u_cost = 0
        for i in range(self.nu):
            u_cost += u[i]**2

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
    
    def cost_fun_BP(self, xt, u):
        x = xt[:self.nx]
        cost = self.cost_fun(x, u)

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

