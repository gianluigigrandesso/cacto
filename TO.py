import sys
import numpy as np
import tensorflow as tf
from pyomo.dae import *
from pyomo.environ import *
from tensorflow.keras import layers, regularizers
from inits import init_action, init_state, init_state_ICS, init_0

class TO_Pyomo:
    '''
    :system_id:                  (str) Id system
    '''

    def __init__(self, env, conf, system_id):

        self.env = env
        self.conf = conf

        self.system_id = system_id

        return

    def TO_Manipulator_Model(self,ICS, init_q0, init_q1, init_q2, init_v0, init_v1, init_v2, init_a0, init_a1, init_a2, init_tau0, init_tau1, init_tau2, N, init_TO=None):
        ''' Create TO pyomo model - manipulator '''
        m = ConcreteModel()
        m.k = RangeSet(0, N)

        if init_TO != None:
            init_TO_controls = init_TO[0]
            init_TO_states = init_TO[1]      
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,0,init_TO_controls), bounds=(self.conf.u_min[0], self.conf.u_max[0])) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,1,init_TO_controls), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.tau2 = Var(m.k, initialize=init_tau2(m,m.k,2,init_TO_controls), bounds=(self.conf.u_min[2], self.conf.u_max[2])) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,0,init_TO_states))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,1,init_TO_states))
            m.q2 = Var(m.k, initialize=init_q2(m,m.k,2,init_TO_states))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,3,init_TO_states))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,4,init_TO_states))
            m.v2 = Var(m.k, initialize=init_v2(m,m.k,5,init_TO_states))
        else:    
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,0), bounds=(self.conf.u_min[0], self.conf.u_max[0]))
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,1), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.tau2 = Var(m.k, initialize=init_tau2(m,m.k,2), bounds=(self.conf.u_min[2], self.conf.u_max[2]))
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,0,ICS))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,1,ICS))
            m.q2 = Var(m.k, initialize=init_q2(m,m.k,2,ICS))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,3,ICS))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,4,ICS))
            m.v2 = Var(m.k, initialize=init_v2(m,m.k,5,ICS))
        
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
           m.v0[k+1] == m.v0[k] + self.conf.dt*m.a0[k] if k < N else Constraint.Skip)

        m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + self.conf.dt*m.a1[k] if k < N else Constraint.Skip)

        m.v2_update = Constraint(m.k, rule = lambda m, k:
           m.v2[k+1] == m.v2[k] + self.conf.dt*m.a2[k] if k < N else Constraint.Skip)

        m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + self.conf.dt*m.v0[k] if k < N else Constraint.Skip)

        m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + self.conf.dt*m.v1[k] if k < N else Constraint.Skip)

        m.q2_update = Constraint(m.k, rule = lambda m, k:
           m.q2[k+1] == m.q2[k] + self.conf.dt*m.v2[k] if k < N else Constraint.Skip)


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
        
        # Rename reward parameters
        alpha = self.conf.soft_max_param[0]
        alpha2 = self.conf.soft_max_param[1]

        XC1 = self.conf.obs_param[0]
        YC1 = self.conf.obs_param[1]
        XC2 = self.conf.obs_param[2]
        YC2 = self.conf.obs_param[3]
        XC3 = self.conf.obs_param[4]
        YC3 = self.conf.obs_param[5]
        
        A1 = self.conf.obs_param[6]
        B1 = self.conf.obs_param[7]
        A2 = self.conf.obs_param[8]
        B2 = self.conf.obs_param[9]
        A3 = self.conf.obs_param[10]
        B3 = self.conf.obs_param[11]

        w_d = self.conf.weight[0]
        w_u = self.conf.weight[1]
        w_peak = self.conf.weight[2]
        w_ob1 = self.conf.weight[3]
        w_ob2 = self.conf.weight[3]
        w_ob3 = self.conf.weight[3]
        w_v = self.conf.weight[4]

        x_des = self.conf.TARGET_STATE[0]
        y_des = self.conf.TARGET_STATE[1]

        ### Penalties representing the obstacle ###
        m.ell1_penalty = sum((log(exp(alpha*-((((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-XC1)**2)/((A1/2)**2) + (((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((self.conf.x_base + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-XC1)**2)/((A1/2)**2) + ((self.conf.y_base + self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)
        m.ell2_penalty = sum((log(exp(alpha*-((((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-XC2)**2)/((A2/2)**2) + (((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((self.conf.x_base + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-XC2)**2)/((A2/2)**2) + ((self.conf.y_base + self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)
        m.ell3_penalty = sum((log(exp(alpha*-((((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-XC3)**2)/((A3/2)**2) + (((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((self.conf.x_base + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-XC3)**2)/((A3/2)**2) + ((self.conf.y_base + self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)

        ### Control effort term ###
        m.u_obj = sum((m.tau0[k]**2 + m.tau1[k]**2 + m.tau2[k]**2) for k in m.k) - (m.tau0[N]**2 + m.tau1[N]**2 + m.tau2[N]**2)
    
        ### Distence to target term (quadratic term + log valley centered at target) ###
        m.dist_cost = sum((w_d*(((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-x_des)**2 + ((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt(((self.conf.x_base + self.conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-x_des)**2 +0.1) - 0.1 + sqrt(((self.conf.y_base + self.conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2)) for k in m.k) - (w_d*((self.conf.x_base + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-x_des)**2 + (self.conf.y_base + self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt((self.conf.x_base + self.conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-x_des)**2 +0.1) - 0.1 + sqrt((self.conf.y_base + self.conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2))

        ### Final velocity term ###
        m.v = (m.v0[N])**2 + (m.v1[N])**2 + (m.v2[N])**2

        m.obj = Objective(expr = (m.dist_cost + w_v*m.v + w_ob1*m.ell1_penalty + w_ob2*m.ell2_penalty + w_ob3*m.ell3_penalty + w_u*m.u_obj - 10000)/100, sense=minimize)
        return m

    def TO_Manipulator_Solve(self, ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH):
        ''' Solve TO problem - manipulator '''
        if ep < self.conf.EPISODE_CRITIC_PRETRAINING or ep < self.conf.EPISODE_ICS_INIT:
            TO_mdl = self.TO_Manipulator_Model(init_rand_state, init_state_ICS, init_state_ICS, init_state_ICS, init_state_ICS, init_state_ICS, init_state_ICS, init_0, init_0, init_0, init_0, init_0, init_0, NSTEPS_SH)
        else:
            TO_mdl = self.TO_Manipulator_Model(init_rand_state, init_state, init_state, init_state, init_state, init_state, init_state, init_0, init_0, init_0, init_action, init_action, init_action, NSTEPS_SH, init_TO = [init_TO_controls, init_TO_states])

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

                success_flag = 1
            else:
                print('TO solution not optimal')     
                raise Exception()         
        except:
            success_flag = 0
            tau_TO = None
            print("*** TO failed ***")  

        x_ee_arr = [(self.conf.x_base + self.conf.l*(cos(TO_mdl.q0[k]) + cos(TO_mdl.q0[k]+TO_mdl.q1[k]) + cos(TO_mdl.q0[k]+TO_mdl.q1[k]+TO_mdl.q2[k])))() for k in K]
        y_ee_arr = [(self.conf.y_base + self.conf.l*(sin(TO_mdl.q0[k]) + sin(TO_mdl.q0[k]+TO_mdl.q1[k]) + sin(TO_mdl.q0[k]+TO_mdl.q1[k]+TO_mdl.q2[k])))() for k in K]

        return success_flag, tau_TO, x_ee_arr, y_ee_arr

    def TO_DoubleIntegrator_Model(self,ICS, init_q0, init_q1, init_v0, init_v1, init_a0, init_a1, init_tau0, init_tau1, N, init_TO=None):
        ''' Create TO pyomo model - double integrator '''
        m = ConcreteModel()
        m.k = RangeSet(0, N)
        
        if init_TO != None:
            init_TO_controls = init_TO[0]
            init_TO_states = init_TO[1]     
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,0,init_TO_controls), bounds=(self.conf.u_min[0], self.conf.u_max[0])) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,1,init_TO_controls), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,0,init_TO_states))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,1,init_TO_states))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,2,init_TO_states))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,3,init_TO_states))
        else:   
            m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,0), bounds=(self.conf.u_min[0], self.conf.u_max[0])) 
            m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,1), bounds=(self.conf.u_min[1], self.conf.u_max[1])) 
            m.q0 = Var(m.k, initialize=init_q0(m,m.k,0,ICS))
            m.q1 = Var(m.k, initialize=init_q1(m,m.k,1,ICS))
            m.v0 = Var(m.k, initialize=init_v0(m,m.k,2,ICS))
            m.v1 = Var(m.k, initialize=init_v1(m,m.k,3,ICS))
        
        m.a0 = Var(m.k, initialize=init_a0)
        m.a1 = Var(m.k, initialize=init_a1)

        m.icfix_q0 = Constraint(rule = lambda m: m.q0[0] == ICS[0])
        m.icfix_q1 = Constraint(rule = lambda m: m.q1[0] == ICS[1])
        m.icfix_v0 = Constraint(rule = lambda m: m.v0[0] == ICS[2])
        m.icfix_v1 = Constraint(rule = lambda m: m.v1[0] == ICS[3])        

        ### Integration ###
        m.v0_update = Constraint(m.k, rule = lambda m, k:
           m.v0[k+1] == m.v0[k] + self.conf.dt*m.a0[k] if k < N else Constraint.Skip)

        m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + self.conf.dt*m.a1[k] if k < N else Constraint.Skip)

        m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + self.conf.dt*m.v0[k] + self.conf.dt**2*m.a0[k]/2 if k < N else Constraint.Skip)

        m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + self.conf.dt*m.v1[k] + self.conf.dt**2*m.a1[k]/2 if k < N else Constraint.Skip)

        # Rename reward parameters
        alpha = self.conf.soft_max_param[0]
        alpha2 = self.conf.soft_max_param[1]

        XC1 = self.conf.obs_param[0]
        YC1 = self.conf.obs_param[1]
        XC2 = self.conf.obs_param[2]
        YC2 = self.conf.obs_param[3]
        XC3 = self.conf.obs_param[4]
        YC3 = self.conf.obs_param[5]
        
        A1 = self.conf.obs_param[6]
        B1 = self.conf.obs_param[7]
        A2 = self.conf.obs_param[8]
        B2 = self.conf.obs_param[9]
        A3 = self.conf.obs_param[10]
        B3 = self.conf.obs_param[11]

        w_d = self.conf.weight[0]
        w_u = self.conf.weight[1]
        w_peak = self.conf.weight[2]
        w_ob1 = self.conf.weight[3]
        w_ob2 = self.conf.weight[3]
        w_ob3 = self.conf.weight[3]
        w_v = self.conf.weight[4]

        x_des = self.conf.TARGET_STATE[0]
        y_des = self.conf.TARGET_STATE[1]

        ### Penalties representing the obstacle ###
        m.ell1_penalty = sum((log(exp(alpha*-(((m.q0[k]-XC1)**2)/((A1/2)**2) + ((m.q1[k]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((m.q0[0]-XC1)**2)/((A1/2)**2) + ((m.q1[0]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha)
        m.ell2_penalty = sum((log(exp(alpha*-(((m.q0[k]-XC2)**2)/((A2/2)**2) + ((m.q1[k]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((m.q0[0]-XC2)**2)/((A2/2)**2) + ((m.q1[0]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha)
        m.ell3_penalty = sum((log(exp(alpha*-(((m.q0[k]-XC3)**2)/((A3/2)**2) + ((m.q1[k]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)  for k in m.k) - (log(exp(alpha*-(((m.q0[0]-XC3)**2)/((A3/2)**2) + ((m.q1[0]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha)

        ### Control effort term ###
        m.u_obj = sum((m.tau0[k]**2 + m.tau1[k]**2) for k in m.k) - (m.tau0[0]**2 + m.tau1[0]**2)

        ### Distence to target term (quadratic term + log valley centered at target) ###
        m.dist_cost = sum((w_d*((m.q0[k]-x_des)**2 + (m.q1[k]-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt((m.q0[k]-x_des)**2 +0.1) - 0.1 + sqrt((m.q1[k]-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2) - 10000) for k in m.k) - (w_d*((m.q0[0]-x_des)**2 + (m.q1[0]-y_des)**2) - w_peak*(log(exp(alpha2*-(sqrt((m.q0[0]-x_des)**2 +0.1) - 0.1 + sqrt((m.q1[0]-y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/alpha2) - 10000)

        m.obj = Objective(expr = (1/100)*(m.dist_cost + w_ob1*m.ell1_penalty + w_ob2*m.ell2_penalty + w_ob3*m.ell3_penalty + w_u*m.u_obj), sense=minimize)
        
        return m

    def TO_DoubleIntegrator_Solve(self, ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH):
        ''' Solve TO problem - double integrator '''               
        if ep < self.conf.EPISODE_CRITIC_PRETRAINING or ep < self.conf.EPISODE_ICS_INIT:
            TO_mdl = self.TO_DoubleIntegrator_Model(init_rand_state, init_state_ICS, init_state_ICS, init_state_ICS, init_state_ICS, init_0, init_0, init_0, init_0, NSTEPS_SH)
        else:
            TO_mdl = self.TO_DoubleIntegrator_Model(init_rand_state, init_state, init_state, init_state, init_state, init_0, init_0, init_action, init_action, NSTEPS_SH, init_TO = [init_TO_controls, init_TO_states])
        
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
                
                success_flag = 1
            else:
                print('TO solution not optimal')    
                raise Exception()         
        except:
            success_flag = 0
            tau_TO = None
            print("*** TO failed ***")  

        x_ee_arr = [TO_mdl.q0[k]() for k in K]
        y_ee_arr = [TO_mdl.q1[k]() for k in K]
        
        return success_flag, tau_TO, x_ee_arr, y_ee_arr


    def TO_Solve(self, ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH):
        ''' Create TO problem '''
        # sys_dep #
        if self.system_id == 'double_integrator':
            success_flag, tau_TO, x_ee_arr, y_ee_arr = self.TO_DoubleIntegrator_Solve(ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH) 
        elif self.system_id == 'manipulator':
            success_flag, tau_TO, x_ee_arr, y_ee_arr = self.TO_Manipulator_Solve(ep, init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
        else:
            print('Pyomo {} model not found'.format(self.system_id))

        # Plot TO solution    
        # plot_results_TO(TO_mdl)  
             
        return tau_TO, success_flag, x_ee_arr, y_ee_arr     
     


class TO_Casadi:
    def __init__(self, env, conf):

        self.env = env
        self.conf = conf

        print('Not implemented')
        sys.exit()