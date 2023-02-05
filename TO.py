import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
from inits import init_tau0,init_tau1,init_tau2,init_q0,init_q1,init_q2,init_v0,init_v1,init_v2,init_q0_ICS,init_q1_ICS,init_q2_ICS,init_v0_ICS,init_v1_ICS,init_v2_ICS,init_0
from CACTO import CACTO
import numpy as np
import random
import math

       
class TO_Pyomo(CACTO):
    def __init__(self, env, conf):
        super(TO_Pyomo, self).__init__(env, conf, init_setup_model=False)

        self.soft_max_param = conf.soft_max_param
        self.obs_param = conf.obs_param
        self.weight = conf.weight
        self.target = conf.TARGET_STATE
        self.NSTEPS = conf.NSTEPS
        self.EPISODE_ICS_INIT = conf.EPISODE_ICS_INIT
        self.LR_SCHEDULE = conf.LR_SCHEDULE
        self.state_norm_arr = conf.state_norm_arr

        return

    # Create TO problem
    def TO_Manipulator_Model(self,ICS, init_q0, init_q1, init_q2, init_v0, init_v1, init_v2, init_a0, init_a1, init_a2, init_tau0, init_tau1, init_tau2, N, init_TO=None):

        m = ConcreteModel()
        m.k = RangeSet(0, N)

        if init_TO != None:
            init_TO_controls = init_TO[0]
            init_TO_states = init_TO[1]      
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,init_TO_controls), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,init_TO_controls), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.tau2 = Var(m.k, initialize=init_tau2(m,m.k,init_TO_controls), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,init_TO_states))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,init_TO_states))
            m.q2 = Var(m.k, initialize=init_q2(m,m.k,init_TO_states))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,init_TO_states))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,init_TO_states))
            m.v2 = Var(m.k, initialize=init_v2(m,m.k,init_TO_states))
        else:    
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.tau2 = Var(m.k, initialize=init_tau2(m,m.k), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,ICS))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,ICS))
            m.q2 = Var(m.k, initialize=init_q2(m,m.k,ICS))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,ICS))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,ICS))
            m.v2 = Var(m.k, initialize=init_v2(m,m.k,ICS))
        
        m.a0 = Var(m.k, initialize=init_a0)
        m.a1 = Var(m.k, initialize=init_a1)
        m.a2 = Var(m.k, initialize=init_a2)

        m.icfix_q0 = Constraint(rule = lambda m: m.q0[0] == ICS[0])
        m.icfix_q1 = Constraint(rule = lambda m: m.q1[0] == ICS[1])
        m.icfix_q2 = Constraint(rule = lambda m: m.q2[0] == ICS[2])
        m.icfix_v0 = Constraint(rule = lambda m: m.v0[0] == ICS[3])        
        m.icfix_v1 = Constraint(rule = lambda m: m.v1[0] == ICS[4])        
        m.icfix_v2 = Constraint(rule = lambda m: m.v2[0] == ICS[5])        

        ### Integration ###
        m.v0_update = Constraint(m.k, rule = lambda m, k:
           m.v0[k+1] == m.v0[k] + self.dt*m.a0[k] if k < N else Constraint.Skip)

        m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + self.dt*m.a1[k] if k < N else Constraint.Skip)

        m.v2_update = Constraint(m.k, rule = lambda m, k:
           m.v2[k+1] == m.v2[k] + self.dt*m.a2[k] if k < N else Constraint.Skip)

        m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + self.dt*m.v0[k] if k < N else Constraint.Skip)

        m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + self.dt*m.v1[k] if k < N else Constraint.Skip)

        m.q2_update = Constraint(m.k, rule = lambda m, k:
           m.q2[k+1] == m.q2[k] + self.dt*m.v2[k] if k < N else Constraint.Skip)


        ### Dyamics ###
        m.EoM_0 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*self.conf.Iz*m.a2[k] + self.conf.l**2*self.conf.M*m.a2[k] + 2*self.conf.l**2*self.conf.M*m.a2[k]*cos(m.q2[k]) + 2*self.conf.l**2*self.conf.M*m.a2[k]*cos(m.q1[k] + m.q2[k]) + 2*m.a1[k]*(4*self.conf.Iz + 3*self.conf.l**2*self.conf.M + 3*self.conf.l**2*self.conf.M*cos(m.q1[k]) + 2*self.conf.l**2*self.conf.M*cos(m.q2[k]) + self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k])) + m.a0[k]*(12*self.conf.Iz + 15*self.conf.l**2*self.conf.M +
    12*self.conf.l**2*self.conf.M*cos(m.q1[k]) + 4*self.conf.l**2*self.conf.M*cos(m.q2[k]) + 4*self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k])) - 4*m.tau0[k] - 12*self.conf.l**2*self.conf.M*sin(m.q1[k])*m.v0[k]*m.v1[k] - 4*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]*m.v1[k] - 6*self.conf.l**2*self.conf.M*sin(m.q1[k])*m.v1[k]**2 - 2*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v1[k]**2
    - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v1[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v1[k]*m.v2[k] - 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v2[k]**2 - 2*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v2[k]**2) == 0 if k < N else Constraint.Skip)

        m.EoM_1 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*self.conf.Iz*m.a2[k] + self.conf.l**2*self.conf.M*m.a2[k] + 2*self.conf.l**2*self.conf.M*m.a2[k]*cos(m.q2[k]) + m.a1[k]*(8*self.conf.Iz + 6*self.conf.l**2*self.conf.M + 4*self.conf.l**2*self.conf.M*cos(m.q2[k])) + 2*m.a0[k]*(4*self.conf.Iz + 3*self.conf.l**2*self.conf.M + 3*self.conf.l**2*self.conf.M*cos(m.q1[k]) + 2*self.conf.l**2*self.conf.M*cos(m.q2[k]) + self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k]))
    - 4*m.tau1[k] + 6*self.conf.l**2*self.conf.M*sin(m.q1[k])*m.v0[k]**2 + 2*self.conf.l**2*self.conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]**2 - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]*m.v2[k] - 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v1[k]*m.v2[k] - 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v2[k]**2) == 0 if k < N else Constraint.Skip)

        m.EoM_2 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*self.conf.Iz*m.a2[k] + self.conf.l**2*self.conf.M*m.a2[k] + m.a1[k]*(4*self.conf.Iz + self.conf.l**2*self.conf.M + 2*self.conf.l**2*self.conf.M*cos(m.q2[k])) + m.a0[k]*(4*self.conf.Iz + self.conf.l**2*self.conf.M + 2*self.conf.l**2*self.conf.M*cos(m.q2[k]) + 2*self.conf.l**2*self.conf.M*cos(m.q1[k] + m.q2[k])) - 4*m.tau2[k] + 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]**2 + 2*self.conf.l**2*self.conf.M*sin(m.q1[k] +
    m.q2[k])*m.v0[k]**2 + 4*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v0[k]*m.v1[k] + 2*self.conf.l**2*self.conf.M*sin(m.q2[k])*m.v1[k]**2) == 0 if k < N else Constraint.Skip)
        
        #rename reward parameters
        alpha = self.soft_max_param[0]
        alpha2 = self.soft_max_param[1]

        XC1 = self.obs_param[0]
        YC1 = self.obs_param[1]
        XC2 = self.obs_param[2]
        YC2 = self.obs_param[3]
        XC3 = self.obs_param[4]
        YC3 = self.obs_param[5]
        
        A1 = self.obs_param[6]
        B1 = self.obs_param[7]
        A2 = self.obs_param[8]
        B2 = self.obs_param[9]
        A3 = self.obs_param[10]
        B3 = self.obs_param[11]

        w_d = self.weight[0]
        w_u = self.weight[1]
        w_peak = self.weight[2]
        w_ob1 = self.weight[3]
        w_ob2 = self.weight[3]
        w_ob3 = self.weight[3]
        w_v = self.weight[4]

        x_des = self.target[0]
        y_des = self.target[1]

        ### Penalties representing the obstacle ###
        m.ell1_penalty = sum((log(exp(alpha*-((((-7 + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-XC1)**2)/((A1/2)**2) + (((self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((-7 + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-XC1)**2)/((A1/2)**2) + ((self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)
        m.ell2_penalty = sum((log(exp(alpha*-((((-7 + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-XC2)**2)/((A2/2)**2) + (((self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((-7 + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-XC2)**2)/((A2/2)**2) + ((self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)
        m.ell3_penalty = sum((log(exp(alpha*-((((-7 + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-XC3)**2)/((A3/2)**2) + (((self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((-7 + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-XC3)**2)/((A3/2)**2) + ((self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)

        ### Control effort term ###
        m.u_obj = sum((m.tau0[k]**2 + m.tau1[k]**2 + m.tau2[k]**2) for k in m.k) - (m.tau0[N]**2 + m.tau1[N]**2 + m.tau2[N]**2)
    
        ### Distence to target term (quadratic term + log valley centered at target) ###
        m.dist_cost = sum((w_d*(((-7 + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-x_des)**2 + ((self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt(((-7 + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-x_des)**2 +0.1) - 0.1 + sqrt(((self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2)) for k in m.k) - (w_d*((-7 + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-x_des)**2 + (self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt((-7 + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-x_des)**2 +0.1) - 0.1 + sqrt((self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2))

        # m.v = sum(((m.v0[k])**2 + (m.v1[k])**2 + (m.v2[k])**2)*(k/m.k[-1])**2 for k in m.k) 
        m.v = (m.v0[N])**2 + (m.v1[N])**2 + (m.v2[N])**2

        m.obj = Objective(expr = (m.dist_cost + w_v*m.v + w_ob1*m.ell1_penalty + w_ob2*m.ell2_penalty + w_ob3*m.ell3_penalty + w_u*m.u_obj - 10000)/100, sense=minimize)
        return m

    def TO_Manipulator_Solve(self, ep, prev_state, init_TO_controls, init_TO_states):
        # Create TO problem                   
        if ep < self.EPISODE_CRITIC_PRETRAINING or ep < self.EPISODE_ICS_INIT:
            TO_mdl = self.TO_DoubleInterator_Model(prev_state, init_q0_ICS, init_q1_ICS, init_v0_ICS, init_v1_ICS, init_0, init_0, init_0, init_0, CACTO.NSTEPS_SH)
        else:
            if ep == self.EPISODE_ICS_INIT and self.LR_SCHEDULE:  
                # Re-initialize Adam otherwise it keeps being affected by the estimates of first-order and second-order moments computed previously with ICS warm-starting
                CACTO.critic_optimizer = tf.keras.optimizers.Adam(CACTO.CRITIC_LR_SCHEDULE)     
                CACTO.actor_optimizer = tf.keras.optimizers.Adam(CACTO.ACTOR_LR_SCHEDULE)
            TO_mdl = self.TO_Manipulator_Model(prev_state, init_q0, init_q1, init_q2, init_v0, init_v1, init_v2, init_0, init_0, init_0, init_tau0, init_tau1, init_tau2, CACTO.NSTEPS_SH, init_TO = [init_TO_controls, init_TO_states])
            
        # Indexes of TO variables       
        K = np.array([k for k in TO_mdl.k]) 

        # Select solver
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = "ma57"

        ### Try to solve TO problem
        try:
            results = solver.solve(TO_mdl)                              
            if str(results.solver.termination_condition) == "optimal":    
                #Retrieve control trajectory
                tau0_TO = [TO_mdl.tau0[k]() for k in K]
                tau1_TO = [TO_mdl.tau1[k]() for k in K]
                tau2_TO = [TO_mdl.tau2[k]() for k in K]
                t0 = np.array(tau0_TO).reshape(len(K),1)
                t1 = np.array(tau1_TO).reshape(len(K),1)
                t2 = np.array(tau2_TO).reshape(len(K),1)
                tau_TO = np.concatenate((t0, t1, t2),axis=1)
            
                TO = 1
            else:
                print('TO solution not optimal')                   
                raise Exception()         
        except:
            print("*** TO failed ***")  
        
        return TO, tau_TO

    def TO_DoubleIntegrator_Model(self,ICS, init_q0, init_q1, init_v0, init_v1, init_a0, init_a1, init_tau0, init_tau1, N, init_TO=None):
        m = ConcreteModel()
        m.k = RangeSet(0, N)
        
        if init_TO != None:
            init_TO_controls = init_TO[0]
            init_TO_states = init_TO[1]     
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,init_TO_controls), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,init_TO_controls), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,init_TO_states))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,init_TO_states))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,init_TO_states))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,init_TO_states))
        else:    
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k), bounds=(-self.tau_upper_bound, self.tau_upper_bound)) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,ICS))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,ICS))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,ICS))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,ICS))
        
        m.a0 = Var(m.k, initialize=init_a0)
        m.a1 = Var(m.k, initialize=init_a1)

        m.icfix_q0 = Constraint(rule = lambda m: m.q0[0] == ICS[0])
        m.icfix_q1 = Constraint(rule = lambda m: m.q1[0] == ICS[1])
        m.icfix_v0 = Constraint(rule = lambda m: m.v0[0] == ICS[2])
        m.icfix_v1 = Constraint(rule = lambda m: m.v1[0] == ICS[3])        

        m.v0_update = Constraint(m.k, rule = lambda m, k:
           m.v0[k+1] == m.v0[k] + self.dt*m.a0[k] if k < N else Constraint.Skip)

        m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + self.dt*m.a1[k] if k < N else Constraint.Skip)

        m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + self.dt*m.v0[k] if k < N else Constraint.Skip)

        m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + self.dt*m.v1[k] if k < N else Constraint.Skip)

        #rename reward parameters
        alpha = self.soft_max_param[0]
        alpha2 = self.soft_max_param[1]

        XC1 = self.obs_param[0]
        YC1 = self.obs_param[1]
        XC2 = self.obs_param[2]
        YC2 = self.obs_param[3]
        XC3 = self.obs_param[4]
        YC3 = self.obs_param[5]
        
        A1 = self.obs_param[6]
        B1 = self.obs_param[7]
        A2 = self.obs_param[8]
        B2 = self.obs_param[9]
        A3 = self.obs_param[10]
        B3 = self.obs_param[11]

        w_d = self.weight[0]
        w_u = self.weight[1]
        w_peak = self.weight[2]
        w_ob1 = self.weight[3]
        w_ob2 = self.weight[3]
        w_ob3 = self.weight[3]
        w_v = self.weight[4]

        x_des = self.target[0]
        y_des = self.target[1]

        m.ell1_penalty = sum((log(exp(alpha*-(((m.q0[k]-XC1)**2)/((A1/2)**2) + ((m.q1[k]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((m.q0[0]-XC1)**2)/((A1/2)**2) + ((m.q1[0]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)
        m.ell2_penalty = sum((log(exp(alpha*-(((m.q0[k]-XC2)**2)/((A2/2)**2) + ((m.q1[k]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((m.q0[0]-XC2)**2)/((A2/2)**2) + ((m.q1[0]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)
        m.ell3_penalty = sum((log(exp(alpha*-(((m.q0[k]-XC3)**2)/((A3/2)**2) + ((m.q1[k]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((m.q0[0]-XC3)**2)/((A3/2)**2) + ((m.q1[0]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)

        m.u_obj = sum((m.tau0[k]**2 + m.tau1[k]**2) for k in m.k) - (m.tau0[0]**2 + m.tau1[0]**2)

        m.dist_cost = sum((w_d*((m.q0[k]-x_des)**2 + (m.q1[k]-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt((m.q0[k]-x_des)**2 +0.1) - 0.1 + sqrt((m.q1[k]-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2) - 10000) for k in m.k) - (w_d*((m.q0[0]-x_des)**2 + (m.q1[0]-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt((m.q0[0]-x_des)**2 +0.1) - 0.1 + sqrt((m.q1[0]-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2) - 10000) #- w_peak*(log(exp(alpha2*-((m.q0[1,k]-x_des) - 0.1 + (m.q0[2,k]-y_des) - 0.1)) + 1)/alpha2)

        m.obj = Objective(expr = (1/100)*(m.dist_cost + w_ob1*m.ell1_penalty + w_ob2*m.ell2_penalty + w_ob3*m.ell3_penalty + w_u*m.u_obj), sense=minimize) #+  w_ob*(m.ell1_penalty+m.ell2_penalty+m.ell3_penalty)
        
        return m

    def TO_DoubleIntegrator_Solve(self, ep, prev_state, init_TO_controls, init_TO_states):
        # Create TO problem               
        if ep < self.EPISODE_CRITIC_PRETRAINING or ep < self.EPISODE_ICS_INIT:
            TO_mdl = self.TO_DoubleInterator_Model(prev_state, init_q0_ICS, init_q1_ICS, init_v0_ICS, init_v1_ICS, init_0, init_0, init_0, init_0, CACTO.NSTEPS_SH)
        else:
            if ep == self.EPISODE_ICS_INIT and self.LR_SCHEDULE:  
                # Re-initialize Adam otherwise it keeps being affected by the estimates of first-order and second-order moments computed previously with ICS warm-starting
                CACTO.critic_optimizer = tf.keras.optimizers.Adam(CACTO.CRITIC_LR_SCHEDULE)     
                CACTO.actor_optimizer = tf.keras.optimizers.Adam(CACTO.ACTOR_LR_SCHEDULE)
            TO_mdl = self.TO_DoubleIntegrator_Model(prev_state, init_q0, init_q1, init_v0, init_v1, init_0, init_0, init_tau0, init_tau1, CACTO.NSTEPS_SH, init_TO = [init_TO_controls, init_TO_states])
            
        # Indexes of TO variables       
        K = np.array([k for k in TO_mdl.k]) 

        # Select solver
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = "ma57"

        ### Try to solve TO problem
        try:
            results = solver.solve(TO_mdl)                              
            if str(results.solver.termination_condition) == "optimal":    
                #Retrieve control trajectory
                tau0_TO = [TO_mdl.tau0[k]() for k in K]
                tau1_TO = [TO_mdl.tau1[k]() for k in K]
                t0 = np.array(tau0_TO).reshape(len(K),1)
                t1 = np.array(tau1_TO).reshape(len(K),1)
                tau_TO = np.concatenate((t0, t1),axis=1)
            
                TO = 1
            else:
                print('TO solution not optimal')                   
                raise Exception()         
        except:
            print("*** TO failed ***")  
        
        return TO, tau_TO


    def TO_Solve(self, ep, env):
        TO = 0             # Flag to indicate if the TO problem has been solved

        # START TO PROBLEM 
        while TO==0:
            
            # Randomize initial state 
            rand_time, prev_state = self.env.reset()
            #rand_time = random.uniform(0,(self.NSTEPS-1)*self.dt)
            #prev_state = np.array([random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), 
            #            random.uniform(-math.pi/4,math.pi/4), random.uniform(-math.pi/4,math.pi/4), random.uniform(-math.pi/4,math.pi/4),
            #            self.dt*round(rand_time/self.dt)]) 
            #prev_state = np.array([random.uniform(-15,15), random.uniform(-15,15),
            #            random.uniform(-6,6), random.uniform(-6,6),
            #            self.dt*round(rand_time/self.dt)])
                        
            if self.NORMALIZE_INPUTS:
                prev_state_norm = prev_state / self.state_norm_arr
                prev_state_norm[-1] = 2*prev_state_norm[-1] - 1
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_norm), 0)   
            else:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)         

            # Set the horizon of TO problem / RL episode 
            CACTO.NSTEPS_SH = self.NSTEPS - int(round(rand_time/self.dt))  

            # Lists to store TO state and control trajectories
            CACTO.control_arr = np.empty((0, self.nb_action))
            CACTO.state_arr = np.array([prev_state])
            
            ###
            CACTO.x_ee_arr = [self.env.get_end_effector_position(CACTO.state_arr[-1,:])[0]]
            CACTO.y_ee_arr = [self.env.get_end_effector_position(CACTO.state_arr[-1,:])[1]] 
            ###
                 
            # Actor rollout used to initialize TO state and control variables
            init_TO_states = np.zeros((self.nb_state, CACTO.NSTEPS_SH+1))
            for i in range(self.robot.nv+self.robot.nv):
                init_TO_states[i][0] = prev_state[i]                    
            init_TO_controls = np.zeros((self.nb_action, CACTO.NSTEPS_SH+1))
            for i in range(self.robot.na):
                init_TO_controls[i][0] = tf.squeeze(CACTO.actor_model(tf_prev_state)).numpy()[i]
            init_prev_state = np.copy(prev_state)

            # Simulate actor's actions to compute the state trajectory used to initialize TO state variables
            for i in range(1, CACTO.NSTEPS_SH+1):    
                init_TO_controls_sim = np.empty(self.nb_action)
                for j in range(self.nb_action):
                    init_TO_controls_sim[j] = init_TO_controls[j][i-1]                                                                                                                               
                init_next_state =  env.simulate(init_prev_state,init_TO_controls_sim)

                for j in range(self.robot.nv + self.robot.nq):
                    init_TO_states[j][i] = init_next_state[j] 
                if self.NORMALIZE_INPUTS:
                    init_next_state_norm = init_next_state / self.state_norm_arr
                    init_next_state_norm[-1] = 2*init_next_state_norm[-1] - 1
                    init_tf_next_state = tf.expand_dims(tf.convert_to_tensor(init_next_state_norm), 0)        
                else:    
                    init_tf_next_state = tf.expand_dims(tf.convert_to_tensor(init_next_state), 0)  
                for j in range(self.robot.na):      
                    init_TO_controls[j][i] = tf.squeeze(CACTO.actor_model(init_tf_next_state)).numpy()[j]
                
                init_prev_state = np.copy(init_next_state)

            TO, tau_TO = self.TO_DoubleIntegrator_Solve(ep, prev_state, init_TO_controls, init_TO_states) #self.TO_DoubleIntegrator_Solve(ep, prev_state, init_TO_controls, init_TO_states) # sys_dep #
            
            # Plot TO solution    
            # plot_results_TO(TO_mdl)   
              
        return rand_time, prev_state, tau_TO                  

class TO_Casadi(CACTO):
    def __init__(self, env, conf):
        super(TO_Pyomo, self).__init__(env, conf,init_setup_model=False)

        print('Not implemented')
        sys.exit()