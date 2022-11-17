import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import pi
import math
from dynamics_manipulator3DoF import RobotWrapper, RobotSimulator, load_urdf
import config_manipulator3DoF_pyomo as conf
from inits import init_tau0,init_tau1,init_tau2,init_q0,init_q1,init_q2,init_v0,init_v1,init_v2,init_q0_ICS,init_q1_ICS,init_q2_ICS,init_v0_ICS,init_v1_ICS,init_v2_ICS,init_0

# Plot EE trajectories of all TO problems related to the set of initial states considered
def plot_results_TO_all(q0_all,q1_all,x_init_all_sim,y_init_all_sim,x_all_sim,y_all_sim):
    fig = plt.figure(figsize=(12,8))
    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1]) 

    K = range(len(q0_all))
    q0,q1,x1,x2,y1,y2 = [],[],[],[],[],[]
    for i in K:
        q0.append(q0_all[i][0])
        q1.append(q1_all[i][0])
        
    x0 = -7.0
    y0 = 0.0
    for k in K:
        x1.append(x0 + conf.l*cos(q0[k]))    
        x2.append(x0 + conf.l*(cos(q0[k]) + cos(q0[k]+q1[k])))
        y1.append(conf.l*sin(q0[k]))
        y2.append(conf.l*(sin(q0[k]) + sin(q0[k]+q1[k])))  

    ax = fig.add_subplot(1, 1, 1)
    for idx in range(len(x_all_sim)):
        ax.plot(x_all_sim[idx], y_all_sim[idx], 'ro', linewidth=1, markersize=1)
        init_states, = ax.plot(-7,0,'tab:orange',marker = 'o', ms = 15, mec = 'r')
        ax.plot(x_init_all_sim[idx], y_init_all_sim[idx], 'mo', linewidth=1, markersize=1, alpha=0.4)
    targ_state, = ax.plot(conf.x_des,conf.y_des,'k*',markersize=15)
    ax.legend([init_states, targ_state], ['Base', 'Target point EE'], fontsize=20, loc='lower right')    
    ax.add_artist(ell1)
    ax.add_artist(ell2) 
    ax.add_artist(ell3)
    for i in K:
        ax.plot([x0,x1[i],x2[i]],[y0,y1[i],y2[i]],'b',marker = 'o', ms = 5, mec = 'r', linewidth=1, alpha=0.8) #xp, yp, 'r--',
        ax.plot([x2[i],x_all_sim[i][0]],[y2[i],y_all_sim[i][0]],'b', linewidth=1, alpha=0.8)
        if abs(x_all_sim[i][0]-x2[i]) >= 1e-5:
            angle = math.degrees(math.atan((y_all_sim[i][0]-y2[i])/(x_all_sim[i][0]-x2[i])))
        elif y_all_sim[i][0]>0:
            angle = -90
        else:
            angle = 90
        if angle<=1e5 and angle>=-1e5 and x_all_sim[i][0]<0:
            angle+=180
        t = mpl.markers.MarkerStyle(marker='$\sqsubset$')
        t._transform = t.get_transform().rotate_deg(angle)      
        ax.plot(x_all_sim[i][0],y_all_sim[i][0],'b', marker=t, ms = 15, linewidth=1) 
    ax.set_xlim([-41, 31])
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.set_xlabel('X [m]',fontsize=20)
    ax.set_ylabel('Y [m]',fontsize=20)
    ax.set_ylim(-35, 35)
    ax.grid(True)

    fig.tight_layout()
    plt.show()

# Plot trajectories of joint angles and velocities of all TO problems related to the set of initial states considered
def plot_results_TO_all_trajs(q0_all,q1_all,q2_all,v0_all,v1_all,v2_all,tau0_all,tau1_all,tau2_all):
    fig = plt.figure(figsize=(12,10))

    K = range(len(q0_all[0]))
    time = [conf.dt*k for k in K]

    ax2 = fig.add_subplot(3, 1, 1)
    for idx in range(len(q0_all)):
        ax2.plot(time, q0_all[idx], 'r', linewidth=1, markersize=1)
        ax2.plot(time, q1_all[idx], 'b', linewidth=1, markersize=1) 
        ax2.plot(time, q2_all[idx], 'g', linewidth=1, markersize=1) 
    ax2.legend(['q0','q1','q2'],fontsize=16)
    ax2.set_ylabel('[rad]',fontsize=20)
    ax2.grid(True)

    ax3 = fig.add_subplot(3, 1, 2)
    for idx in range(len(q0_all)):
        ax3.plot(time, v0_all[idx], 'r', linewidth=1, markersize=1) 
        ax3.plot(time, v1_all[idx], 'b', linewidth=1, markersize=1) 
        ax3.plot(time, v2_all[idx], 'g', linewidth=1, markersize=1) 
    ax3.legend(['v0','v1','v2'],fontsize=16)
    ax3.set_ylabel('[rad/s]',fontsize=20)
    ax3.grid(True)

    ax4 = fig.add_subplot(3, 1, 3)
    for idx in range(len(q0_all)):
        ax4.plot(time, tau0_all[idx], 'r', linewidth=1, markersize=1)
        ax4.plot(time, tau1_all[idx], 'b', linewidth=1, markersize=1)
        ax4.plot(time, tau2_all[idx], 'g', linewidth=1, markersize=1)
    ax4.legend(['tau0','tau1','tau2'],fontsize=16) 
    ax4.set_ylabel('[Nm]',fontsize=20)
    ax4.set_xlabel('[s]',fontsize=20)
    ax4.grid(True)

    fig.tight_layout()
    plt.show()

# Plot result of one TO problem
def plot_results_TO(m):
    K = np.array([k for k in m.k])  
    tau0 = [m.tau0[k]() for k in K]
    tau1 = [m.tau1[k]() for k in K]
    tau2 = [m.tau2[k]() for k in K]
    x_ee = [(-7 + conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))() for k in K]
    y_ee = [(conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))() for k in K]
    q0 = [m.q0[k]() for k in K]
    q1 = [m.q1[k]() for k in K]
    q2 = [m.q2[k]() for k in K]
    v0 = [m.v0[k]() for k in K]
    v1 = [m.v1[k]() for k in K]
    v2 = [m.v2[k]() for k in K]

    x0 = -7.0
    x1 = [x0 + conf.l*cos(q0[k]) for k in K]    
    x2 = [x0 + conf.l*(cos(q0[k]) + cos(q0[k]+q1[k])) for k in K]
    y0 = 0.0
    y1 = [conf.l*sin(q0[k]) for k in K]
    y2 = [conf.l*(sin(q0[k]) + sin(q0[k]+q1[k])) for k in K]
 
    cost=m.obj()

    fig = plt.figure(figsize=(12,10))
    plt.suptitle('Discrete model', y=1, fontsize=16)

    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(conf.dt*K, x_ee, 'ro', linewidth=1, markersize=1) 
    ax1.plot(conf.dt*K, y_ee, 'bo', linewidth=1, markersize=1) 
    ax1.set_title('Position',fontsize=16)
    ax1.legend(['x','y'],fontsize=16)
    ax1.set_ylim(-31, 31)
    ax1.grid(True)

    ax2 = fig.add_subplot(4, 2, 3)
    ax2.plot(conf.dt*K, q0, 'ro', linewidth=1, markersize=1)
    ax2.plot(conf.dt*K, q1, 'bo', linewidth=1, markersize=1) 
    ax2.plot(conf.dt*K, q2, 'go', linewidth=1, markersize=1)
    ax2.set_title('Joint position',fontsize=16)
    ax2.legend(['q0','q1','q2'],fontsize=16)
    ax2.grid(True)

    ax3 = fig.add_subplot(4, 2, 5)
    ax3.plot(conf.dt*K, v0, 'ro', linewidth=1, markersize=1) 
    ax3.plot(conf.dt*K, v1, 'bo', linewidth=1, markersize=1) 
    ax3.plot(conf.dt*K, v2, 'go', linewidth=1, markersize=1) 
    ax3.set_title('Joint velocity',fontsize=16)
    ax3.legend(['v0','v1','v2'],fontsize=16)
    ax3.grid(True)

    ax4 = fig.add_subplot(4, 2, 7)
    ax4.plot(conf.dt*K, tau0, 'ro', linewidth=1, markersize=1) 
    ax4.plot(conf.dt*K, tau1, 'bo', linewidth=1, markersize=1) 
    ax4.plot(conf.dt*K, tau2, 'go', linewidth=1, markersize=1)
    ax4.legend(['tau0','tau1','tau2'],fontsize=16) 
    ax4.set_title('Controls',fontsize=16)

    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.plot(x_ee, y_ee, 'ro', linewidth=1, markersize=2)
    N_points_conf = 3
    for g in range(N_points_conf):
        index = int(g*K[-1]/(N_points_conf-1))
        if g==0:
            ax3.plot([x0,x1[index],x2[index],x_ee[index]],[y0,y1[index],y2[index],y_ee[index]],'b',marker = 'o', ms = 5, mec = 'r', linewidth=1, alpha=0.8) #xp, yp, 'r--',
        elif g==1:
            ax3.plot([x0,x1[index],x2[index],x_ee[index]],[y0,y1[index],y2[index],y_ee[index]],'dimgrey',marker = 'o', ms = 5, mec = 'r', linewidth=1, alpha=0.8) #xp, yp, 'r--',
        else:
            ax3.plot([x0,x1[index],x2[index],x_ee[index]],[y0,y1[index],y2[index],y_ee[index]],'orange',marker = 'o', ms = 5, mec = 'r', linewidth=1, alpha=0.8) #xp, yp, 'r--',
    ax3.add_artist(ell1)
    ax3.add_artist(ell2) 
    ax3.add_artist(ell3) 
    ax3.set_xlim(-41, 31)
    ax3.set_aspect('equal', 'box')
    ax3.set_title("Plane, OBJ = {}".format(cost),fontsize=16)
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_ylim(-35, 35)

    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    fig.tight_layout()
    plt.show()

# Create TO problem
def TO_manipulator(ICS, init_q0, init_q1, init_q2, init_v0, init_v1, init_v2, init_a0, init_a1, init_a2, init_tau0, init_tau1, init_tau2, N, init_TO=None):
    m = ConcreteModel()
    m.k = RangeSet(0, N)

    if init_TO != None:
        init_TO_controls = init_TO[0]
        init_TO_states = init_TO[1]        
        m.tau0 = Var(m.k, initialize=init_tau0(m,m.k,init_TO_controls), bounds=(-conf.tau_upper_bound, conf.tau_upper_bound)) 
        m.tau1 = Var(m.k, initialize=init_tau1(m,m.k,init_TO_controls), bounds=(-conf.tau_upper_bound, conf.tau_upper_bound)) 
        m.tau2 = Var(m.k, initialize=init_tau2(m,m.k,init_TO_controls), bounds=(-conf.tau_upper_bound, conf.tau_upper_bound)) 
        m.q0 = Var(m.k, initialize=init_q0(m,m.k,init_TO_states))
        m.q1 = Var(m.k, initialize=init_q1(m,m.k,init_TO_states))
        m.q2 = Var(m.k, initialize=init_q2(m,m.k,init_TO_states))
        m.v0 = Var(m.k, initialize=init_v0(m,m.k,init_TO_states))
        m.v1 = Var(m.k, initialize=init_v1(m,m.k,init_TO_states))
        m.v2 = Var(m.k, initialize=init_v2(m,m.k,init_TO_states))
    else:    
        m.tau0 = Var(m.k, initialize=init_tau0(m,m.k), bounds=(-conf.tau_upper_bound, conf.tau_upper_bound)) 
        m.tau1 = Var(m.k, initialize=init_tau1(m,m.k), bounds=(-conf.tau_upper_bound, conf.tau_upper_bound)) 
        m.tau2 = Var(m.k, initialize=init_tau2(m,m.k), bounds=(-conf.tau_upper_bound, conf.tau_upper_bound)) 
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
           m.v0[k+1] == m.v0[k] + conf.dt*m.a0[k] if k < N else Constraint.Skip)

    m.v1_update = Constraint(m.k, rule = lambda m, k:
           m.v1[k+1] == m.v1[k] + conf.dt*m.a1[k] if k < N else Constraint.Skip)

    m.v2_update = Constraint(m.k, rule = lambda m, k:
           m.v2[k+1] == m.v2[k] + conf.dt*m.a2[k] if k < N else Constraint.Skip)

    m.q0_update = Constraint(m.k, rule = lambda m, k:
           m.q0[k+1] == m.q0[k] + conf.dt*m.v0[k] if k < N else Constraint.Skip)

    m.q1_update = Constraint(m.k, rule = lambda m, k:
           m.q1[k+1] == m.q1[k] + conf.dt*m.v1[k] if k < N else Constraint.Skip)

    m.q2_update = Constraint(m.k, rule = lambda m, k:
           m.q2[k+1] == m.q2[k] + conf.dt*m.v2[k] if k < N else Constraint.Skip)


    ### Dyamics ###
    m.EoM_0 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*conf.Iz*m.a2[k] + conf.l**2*conf.M*m.a2[k] + 2*conf.l**2*conf.M*m.a2[k]*cos(m.q2[k]) + 2*conf.l**2*conf.M*m.a2[k]*cos(m.q1[k] + m.q2[k]) + 2*m.a1[k]*(4*conf.Iz + 3*conf.l**2*conf.M + 3*conf.l**2*conf.M*cos(m.q1[k]) + 2*conf.l**2*conf.M*cos(m.q2[k]) + conf.l**2*conf.M*cos(m.q1[k] + m.q2[k])) + m.a0[k]*(12*conf.Iz + 15*conf.l**2*conf.M +
    12*conf.l**2*conf.M*cos(m.q1[k]) + 4*conf.l**2*conf.M*cos(m.q2[k]) + 4*conf.l**2*conf.M*cos(m.q1[k] + m.q2[k])) - 4*m.tau0[k] - 12*conf.l**2*conf.M*sin(m.q1[k])*m.v0[k]*m.v1[k] - 4*conf.l**2*conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]*m.v1[k] - 6*conf.l**2*conf.M*sin(m.q1[k])*m.v1[k]**2 - 2*conf.l**2*conf.M*sin(m.q1[k] + m.q2[k])*m.v1[k]**2
    - 4*conf.l**2*conf.M*sin(m.q2[k])*m.v0[k]*m.v2[k] - 4*conf.l**2*conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]*m.v2[k] - 4*conf.l**2*conf.M*sin(m.q2[k])*m.v1[k]*m.v2[k] - 4*conf.l**2*conf.M*sin(m.q1[k] + m.q2[k])*m.v1[k]*m.v2[k] - 2*conf.l**2*conf.M*sin(m.q2[k])*m.v2[k]**2 - 2*conf.l**2*conf.M*sin(m.q1[k] + m.q2[k])*m.v2[k]**2) == 0 if k < N else Constraint.Skip)

    m.EoM_1 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*conf.Iz*m.a2[k] + conf.l**2*conf.M*m.a2[k] + 2*conf.l**2*conf.M*m.a2[k]*cos(m.q2[k]) + m.a1[k]*(8*conf.Iz + 6*conf.l**2*conf.M + 4*conf.l**2*conf.M*cos(m.q2[k])) + 2*m.a0[k]*(4*conf.Iz + 3*conf.l**2*conf.M + 3*conf.l**2*conf.M*cos(m.q1[k]) + 2*conf.l**2*conf.M*cos(m.q2[k]) + conf.l**2*conf.M*cos(m.q1[k] + m.q2[k]))
    - 4*m.tau1[k] + 6*conf.l**2*conf.M*sin(m.q1[k])*m.v0[k]**2 + 2*conf.l**2*conf.M*sin(m.q1[k] + m.q2[k])*m.v0[k]**2 - 4*conf.l**2*conf.M*sin(m.q2[k])*m.v0[k]*m.v2[k] - 4*conf.l**2*conf.M*sin(m.q2[k])*m.v1[k]*m.v2[k] - 2*conf.l**2*conf.M*sin(m.q2[k])*m.v2[k]**2) == 0 if k < N else Constraint.Skip)

    m.EoM_2 = Constraint(m.k, rule = lambda m, k:
    (1/4)*(4*conf.Iz*m.a2[k] + conf.l**2*conf.M*m.a2[k] + m.a1[k]*(4*conf.Iz + conf.l**2*conf.M + 2*conf.l**2*conf.M*cos(m.q2[k])) + m.a0[k]*(4*conf.Iz + conf.l**2*conf.M + 2*conf.l**2*conf.M*cos(m.q2[k]) + 2*conf.l**2*conf.M*cos(m.q1[k] + m.q2[k])) - 4*m.tau2[k] + 2*conf.l**2*conf.M*sin(m.q2[k])*m.v0[k]**2 + 2*conf.l**2*conf.M*sin(m.q1[k] +
    m.q2[k])*m.v0[k]**2 + 4*conf.l**2*conf.M*sin(m.q2[k])*m.v0[k]*m.v1[k] + 2*conf.l**2*conf.M*sin(m.q2[k])*m.v1[k]**2) == 0 if k < N else Constraint.Skip)

    ### Penalties representing the obstacle ###
    m.ell1_penalty = sum((log(exp(conf.alpha*-((((-7 + conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-conf.XC1)**2)/((conf.A1/2)**2) + (((conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-conf.YC1)**2)/((conf.B1/2)**2) - 1.0)) + 1)/conf.alpha)  for k in m.k) - (log(exp(conf.alpha*-(((-7 + conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-conf.XC1)**2)/((conf.A1/2)**2) + ((conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-conf.YC1)**2)/((conf.B1/2)**2) - 1.0)) + 1)/conf.alpha)
    m.ell2_penalty = sum((log(exp(conf.alpha*-((((-7 + conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-conf.XC2)**2)/((conf.A2/2)**2) + (((conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-conf.YC2)**2)/((conf.B2/2)**2) - 1.0)) + 1)/conf.alpha) for k in m.k) - (log(exp(conf.alpha*-(((-7 + conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-conf.XC2)**2)/((conf.A2/2)**2) + ((conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-conf.YC2)**2)/((conf.B2/2)**2) - 1.0)) + 1)/conf.alpha)
    m.ell3_penalty = sum((log(exp(conf.alpha*-((((-7 + conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-conf.XC3)**2)/((conf.A3/2)**2) + (((conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-conf.YC3)**2)/((conf.B3/2)**2) - 1.0)) + 1)/conf.alpha)  for k in m.k) - (log(exp(conf.alpha*-(((-7 + conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-conf.XC3)**2)/((conf.A3/2)**2) + ((conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-conf.YC3)**2)/((conf.B3/2)**2) - 1.0)) + 1)/conf.alpha)

    ### Control effort term ###
    m.u_obj = sum((m.tau0[k]**2 + m.tau1[k]**2 + m.tau2[k]**2) for k in m.k) - (m.tau0[N]**2 + m.tau1[N]**2 + m.tau2[N]**2)
    
    ### Distence to target term (quadratic term + log valley centered at target) ###
    m.dist_cost = sum((conf.w_d*(((-7 + conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-conf.x_des)**2 + ((conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-conf.y_des)**2) - conf.w_peak*(log(exp(conf.alpha2*-(sqrt(((-7 + conf.l*(cos(m.q0[k]) + cos(m.q0[k]+m.q1[k]) + cos(m.q0[k]+m.q1[k]+m.q2[k])))-conf.x_des)**2 +0.1) - 0.1 + sqrt(((conf.l*(sin(m.q0[k]) + sin(m.q0[k]+m.q1[k]) + sin(m.q0[k]+m.q1[k]+m.q2[k])))-conf.y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/conf.alpha2)) for k in m.k) - (conf.w_d*((-7 + conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-conf.x_des)**2 + (conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-conf.y_des)**2) - conf.w_peak*(log(exp(conf.alpha2*-(sqrt((-7 + conf.l*(cos(m.q0[0]) + cos(m.q0[0]+m.q1[0]) + cos(m.q0[0]+m.q1[0]+m.q2[0]))-conf.x_des)**2 +0.1) - 0.1 + sqrt((conf.l*(sin(m.q0[0]) + sin(m.q0[0]+m.q1[0]) + sin(m.q0[0]+m.q1[0]+m.q2[0]))-conf.y_des)**2 +0.1) - 0.1 -2*sqrt(0.1))) + 1)/conf.alpha2))

    # m.v = sum(((m.v0[k])**2 + (m.v1[k])**2 + (m.v2[k])**2)*(k/m.k[-1])**2 for k in m.k) 
    m.v = (m.v0[N])**2 + (m.v1[N])**2 + (m.v2[N])**2

    m.obj = Objective(expr = (m.dist_cost + conf.w_v*m.v + conf.w_ob1*m.ell1_penalty + conf.w_ob2*m.ell2_penalty + conf.w_ob3*m.ell3_penalty + conf.w_u*m.u_obj - 10000)/100, sense=minimize)
    return m

# Simulate dynamics
def simulate(dt,state,u):

    # Create robot model in Pinocchio with q_init as initial configuration and v_init as initial velocities
    q_init = np.zeros(int((len(state)-1)/2))
    for state_index in range(int((len(state)-1)/2)):
        q_init[state_index] = state[state_index]
    q_init = q_init.T
    v_init = np.zeros(int((len(state)-1)/2))
    for state_index in range(int((len(state)-1)/2)):
        v_init[state_index] = state[int((len(state)-1)/2)+state_index]
    v_init = v_init.T    
    r = load_urdf(URDF_PATH)
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max)

    # Simulate control u
    simu.simulate(u, dt, 1)
    q_next, v_next, a_next = np.copy(simu.q), np.copy(simu.v), np.copy(simu.dv)
    q0_next, q1_next, q2_next = q_next[0],q_next[1],q_next[2]
    v0_next, v1_next, v2_next = v_next[0],v_next[1],v_next[2]

    t_next = state[-1] + dt

    return np.array([q0_next,q1_next,q2_next,v0_next,v1_next,v2_next,t_next])

if __name__ == "__main__":
    from CACTO_manipulator3DoF_pyomo import get_actor
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22 

    N = conf.NSTEPS                         # Number of timesteps                           
    num_states = 7                          # Number of states
    num_actions = 3                         # Number of actions
    q_norm = 15                             # Joint angle normalization
    v_norm = 10                             # Velocity normalization

    simulate_coulomb_friction = 0           # To simulate friction
    simulation_type = 'euler'               # Either 'timestepping' or 'euler'
    tau_coulomb_max = 0*np.ones(3)          # Expressed as percentage of torque max


    # Path to the urdf file
    URDF_PATH = "urdf/planar_manipulator_3dof.urdf"

    ## Inverse kinematics considering the EE being in ICS_ee
    # init_states_sim = []
    # ICS_ee_list = [[10,0],[-10,0],[0,10],[0,-10],[7,7],[-7,7],[7,-7],[-7,-7],[6,0]]
    # for i in range(len(ICS_ee_list)):
    #     ICS_ee = ICS_ee_list[i]
    #     phi = math.atan2(ICS_ee[1],(ICS_ee[0]+7))                           # SUM OF JOINT ANGLES FIXED = orientation of the segment connecting the base with the EE  
    #     X3rd_joint = (ICS_ee[0]+7) - conf.l* math.cos(phi) 
    #     Y3rd_joint = (ICS_ee[1]) - conf.l* math.sin(phi)
    #     c2 = (X3rd_joint**2 + Y3rd_joint**2 -2*conf.l**2)/(2*conf.l**2)
    #     if ICS_ee[1]>=0:
    #         s2 = math.sqrt(1-c2**2)
    #     else:
    #         s2 = -math.sqrt(1-c2**2)
    #     s1 = ((conf.l + conf.l*c2)*Y3rd_joint - conf.l*s2*X3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)  
    #     c1 = ((conf.l + conf.l*c2)*X3rd_joint - conf.l*s2*Y3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)
    #     ICS_q0 = math.atan2(s1,c1)
    #     ICS_q1 = math.atan2(s2,c2)
    #     init_states_sim.append(np.array([ICS_q0,ICS_q1,phi-ICS_q0-ICS_q1,0.0,0.0,0.0,0.0]))   # Initial state

    obj_arr = np.array([])

    # Set of initial states
    # init_states_sim = [np.array([random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), 
                    # random.uniform(-math.pi/4,math.pi/4), random.uniform(-math.pi/4,math.pi/4), random.uniform(-math.pi/4,math.pi/4)])]              
    init_states_sim = [np.array([pi/4,-pi/8,-pi/8,0.0,0.0,0.0,0.0]),np.array([-pi/4,pi/8,pi/8,0.0,0.0,0.0,0.0]),np.array([pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),
                        np.array([-pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([3*pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-3*pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),
                        np.array([pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([pi,0.0,0.0,0.0,0.0,0.0,0.0])]

    # Lists to store TO solutions
    x_ee_all_sim, y_ee_all_sim, q0_all_sim, q1_all_sim, q2_all_sim, v0_all_sim, v1_all_sim, v2_all_sim, a0_all_sim, a1_all_sim, a2_all_sim, tau0_all_sim, tau1_all_sim, tau2_all_sim = [],[],[],[],[],[],[],[],[],[],[],[],[],[]    
    x_ee_init_all_sim, y_ee_init_all_sim, q0_init_all_sim, q1_init_all_sim, q2_init_all_sim, v0_init_all_sim, v1_init_all_sim, v2_init_all_sim, a0_init_all_sim, a1_init_all_sim, a2_init_all_sim = [],[],[],[],[],[],[],[],[],[],[]

    CACTO_ROLLOUT = 1                                                   # Flag to use CACTO warm-start
    N_try = 1                                                           # Number of CACTO training whose policy is going to be rolled-out
    NORMALIZE_INPUTS = 1                                                # If states were normalized during training
    state_norm_arr = np.array([q_norm,q_norm,q_norm,
                                v_norm,v_norm,v_norm,int(N*conf.dt)])   
    
    if CACTO_ROLLOUT:
        NNs_path = './Results/NNs/N_try_{}'.format(N_try)
        
        actor_model = get_actor()

        ## If loading final weights
        # actor_model.load_weights(NNs_path+"Manipulator3DoF_final_actor.h5")

        ## If loading weights saved before completing the training
        nupdates = 128000
        actor_model.load_weights(NNs_path+"/Manipulator_{}.h5".format(nupdates))


    # START SOLVING TO PROBLEMS
    for i in range(len(init_states_sim)):

        ICS = init_states_sim[i]

        if CACTO_ROLLOUT:
            # POLICY ROLLOUT
            q0_CACTO = [init_states_sim[i][0]]
            q1_CACTO = [init_states_sim[i][1]]
            q2_CACTO = [init_states_sim[i][2]]
            v0_CACTO = [init_states_sim[i][3]]
            v1_CACTO = [init_states_sim[i][4]]
            v2_CACTO = [init_states_sim[i][5]]
            a0_CACTO = [0.0]
            a1_CACTO = [0.0]
            a2_CACTO = [0.0]
            tau0_CACTO = []
            tau1_CACTO = []
            tau2_CACTO = []
            step_counter_local = 0
            prev_state_local = np.copy(init_states_sim[i])

            if NORMALIZE_INPUTS:    
                prev_state_norm = prev_state_local / state_norm_arr
                prev_state_norm[-1] = 2*prev_state_norm[-1] - 1                                 
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_norm), 0)
            else:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_local), 0)

            init_TO_states = np.zeros((num_states, N+1))
            init_TO_states[0][0] = prev_state_local[0]
            init_TO_states[1][0] = prev_state_local[1]
            init_TO_states[2][0] = prev_state_local[2]
            init_TO_states[3][0] = prev_state_local[3]                     
            init_TO_states[4][0] = prev_state_local[4]                     
            init_TO_states[5][0] = prev_state_local[5]   
            init_TO_controls = np.zeros((num_actions, N+1))
            init_TO_controls[0][0] = tf.squeeze(actor_model(tf_prev_state)).numpy()[0]
            init_TO_controls[1][0] = tf.squeeze(actor_model(tf_prev_state)).numpy()[1] 
            init_TO_controls[2][0] = tf.squeeze(actor_model(tf_prev_state)).numpy()[2] 
            init_prev_state = np.copy(prev_state_local)

            for i in range(1, N+1):
                init_next_state =  simulate(conf.dt,init_prev_state,np.array([init_TO_controls[0][i-1],init_TO_controls[1][i-1],init_TO_controls[2][i-1]]))
                init_TO_states[0][i] = init_next_state[0]
                init_TO_states[1][i] = init_next_state[1]
                init_TO_states[2][i] = init_next_state[2]
                init_TO_states[3][i] = init_next_state[3] 
                init_TO_states[4][i] = init_next_state[4] 
                init_TO_states[5][i] = init_next_state[5] 
                if NORMALIZE_INPUTS:
                    init_next_state_norm = init_next_state / state_norm_arr
                    init_next_state_norm[-1] = 2*init_next_state_norm[-1] - 1
                    init_tf_next_state = tf.expand_dims(tf.convert_to_tensor(init_next_state_norm), 0)        
                else:    
                    init_tf_next_state = tf.expand_dims(tf.convert_to_tensor(init_next_state), 0)        
                init_TO_controls[0][i] = tf.squeeze(actor_model(init_tf_next_state)).numpy()[0]
                init_TO_controls[1][i] = tf.squeeze(actor_model(init_tf_next_state)).numpy()[1] 
                init_TO_controls[2][i] = tf.squeeze(actor_model(init_tf_next_state)).numpy()[2] 
                init_prev_state = np.copy(init_next_state)

            mdl = TO_manipulator(ICS, init_q0, init_q1, init_q2, init_v0, init_v1, init_v2, init_0, init_0, init_0, init_tau0, init_tau1, init_tau2, N, init_TO = [init_TO_controls, init_TO_states])
        else:
            # Create TO problem choosing how to warm-start the variables between 1)ICS; 2)0s; 3)Random
            mdl = TO_manipulator(ICS, init_q0_ICS, init_q1_ICS, init_q2_ICS, init_v0_ICS, init_v1_ICS, init_v2_ICS, init_0, init_0, init_0, init_0, init_0, init_0, N)
            # mdl = TO_manipulator(ICS, init_0, init_0, init_0, init_0, init_0, init_0, init_0, init_0, init_0, init_0, init_0, init_0, conf.tau_upper_bound, conf.Iz, conf.M, conf.l, conf.w_d, conf.w_v, conf.w_peak, conf.w_ob1, conf.w_ob2, conf.w_ob3, conf.w_u, conf.x_des, conf.y_des, conf.XC1, conf.YC1, conf.XC2, conf.YC2, conf.XC3, conf.YC3, conf.A1, conf.B1, conf.A2, conf.B2, conf.A3, conf.B3, N, conf.dt, conf.alpha, conf.alpha2)
            # mdl = TO_manipulator(ICS, init_rand_q0, init_rand_q1, init_rand_q2, init_rand_v0, init_rand_v1, init_rand_v2, init_rand_a0, init_rand_a1, init_rand_a2, init_rand_tau, init_rand_tau, init_rand_tau, conf.tau_upper_bound, conf.Iz, conf.M, conf.l, conf.w_d, conf.w_v, conf.w_peak, conf.w_ob1, conf.w_ob2, conf.w_ob3, conf.w_u,conf.x_des, conf.y_des, conf.XC1, conf.YC1, conf.XC2, conf.YC2, conf.XC3, conf.YC3, conf.A1, conf.B1, conf.A2, conf.B2, conf.A3, conf.B3, N, conf.dt, conf.alpha, conf.alpha2)

        # Lists storing values to be plotted
        K = np.array([k for k in mdl.k])  
        x_ee_init = [-7+conf.l*(cos(mdl.q0[k]()) + cos(mdl.q0[k]()+mdl.q1[k]()) + cos(mdl.q0[k]()+mdl.q1[k]()+mdl.q2[k]())) for k in K]
        y_ee_init = [(conf.l*(sin(mdl.q0[k]()) + sin(mdl.q0[k]()+mdl.q1[k]()) + sin(mdl.q0[k]()+mdl.q1[k]()+mdl.q2[k]()))) for k in K]
        q0_init = [mdl.q0[k]() for k in K]
        q1_init = [mdl.q1[k]() for k in K]
        q2_init = [mdl.q2[k]() for k in K]
        v0_init = [mdl.v0[k]() for k in K]
        v1_init = [mdl.v1[k]() for k in K]
        v2_init = [mdl.v2[k]() for k in K]
        a0_init = [mdl.a0[k]() for k in K]
        a1_init = [mdl.a1[k]() for k in K]
        a2_init = [mdl.a2[k]() for k in K]
        x_ee_init_all_sim.append(x_ee_init)
        y_ee_init_all_sim.append(y_ee_init)
        q0_init_all_sim.append(q0_init)
        q1_init_all_sim.append(q1_init)
        q2_init_all_sim.append(q2_init)
        v0_init_all_sim.append(v0_init)
        v1_init_all_sim.append(v1_init)
        v2_init_all_sim.append(v2_init)
        a0_init_all_sim.append(a0_init)
        a1_init_all_sim.append(a1_init)
        a2_init_all_sim.append(a2_init)

        # Solve TO problem
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = "ma57"
        try:
            results = solver.solve(mdl,tee=True)    
            print('***** OBJ = ',mdl.obj(),'*****')
            if str(results.solver.termination_condition) != "optimal":
                print(results.solver.termination_condition)
                continue
            else:
                obj_arr = np.append(obj_arr,mdl.obj())
        except:
            print("***** IPOPT FAILED *****")
            continue

        x_ee = [-7+conf.l*(cos(mdl.q0[k]()) + cos(mdl.q0[k]()+mdl.q1[k]()) + cos(mdl.q0[k]()+mdl.q1[k]()+mdl.q2[k]())) for k in K]
        y_ee = [(conf.l*(sin(mdl.q0[k]()) + sin(mdl.q0[k]()+mdl.q1[k]()) + sin(mdl.q0[k]()+mdl.q1[k]()+mdl.q2[k]()))) for k in K]
        q0 = [mdl.q0[k]() for k in K]
        q1 = [mdl.q1[k]() for k in K]
        q2 = [mdl.q2[k]() for k in K]
        v0 = [mdl.v0[k]() for k in K]            
        v1 = [mdl.v1[k]() for k in K]            
        v2 = [mdl.v2[k]() for k in K]            
        a0 = [mdl.a0[k]() for k in K] 
        a1 = [mdl.a1[k]() for k in K] 
        a2 = [mdl.a2[k]() for k in K] 
        tau0 = [mdl.tau0[k]() for k in K]            
        tau1 = [mdl.tau1[k]() for k in K]            
        tau2 = [mdl.tau2[k]() for k in K]            
        x_ee_all_sim.append(x_ee)
        y_ee_all_sim.append(y_ee)
        q0_all_sim.append(q0)
        q1_all_sim.append(q1)
        q2_all_sim.append(q2)
        v0_all_sim.append(v0)        
        v1_all_sim.append(v1)        
        v2_all_sim.append(v2)        
        a0_all_sim.append(a0)        
        a1_all_sim.append(a1)        
        a2_all_sim.append(a2)
        tau0_all_sim.append(tau0)
        tau1_all_sim.append(tau1)
        tau2_all_sim.append(tau2)

    print(obj_arr)

    ## Print trajectories
    # print("sol_tau0 = ",mdl.tau0[:](),"\nsol_tau1 = ",mdl.tau1[:](),"\nsol_tau2 = ",mdl.tau2[:](),"\nsol_q0 = ",mdl.q0[:](),"\nsol_q1 = ",mdl.q1[:](),"\nsol_q2 = ",mdl.q2[:](),"\nsol_v0 = ",mdl.v0[:](),"\nsol_v1 = ",mdl.v1[:](),"\nsol_v2 = ",mdl.v2[:](),"\nsol_a0 = ",mdl.a0[:](),"\nsol_a1 = ",mdl.a1[:](),"\nsol_a2 = ",mdl.a2[:]())
    
    # Plot stuff
    plot_results_TO_all(q0_all_sim,q1_all_sim,x_ee_init_all_sim,y_ee_init_all_sim,x_ee_all_sim,y_ee_all_sim)
    # plot_results_TO_all_trajs(q0_all_sim,q1_all_sim,q2_all_sim,v0_all_sim,v1_all_sim,v2_all_sim,tau0_all_sim,tau1_all_sim,tau2_all_sim)
    # plot_results_TO(mdl) 

    # pickle.dump({"q0":q0_all_sim,"q1":q1_all_sim,"q2":q2_all_sim,"v0":v0_all_sim,"v1":v1_all_sim,"v2":v2_all_sim,"tau0":tau0_all_sim,"tau1":tau1_all_sim,"tau2":tau2_all_sim},open('9_trajectories_ICSinit', 'wb'))