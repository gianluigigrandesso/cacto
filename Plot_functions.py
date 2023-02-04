import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from pyomo.environ import *
from pyomo.dae import *
import manipulator_conf as conf
import manipulator_conf as conf
#import os
#import tensorflow as tf
#from tensorflow.keras import layers, regularizers
#import pinocchio as pin
#import random
#from replay_buffer import PrioritizedReplayBuffer
#from inits import init_tau0,init_tau1,init_tau2,init_q0,init_q1,init_q2,init_v0,init_v1,init_v2,init_q0_ICS,init_q1_ICS,init_q2_ICS,init_v0_ICS,init_v1_ICS,init_v2_ICS,init_0



# os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Uncomment to run TF on CPU rather than GPU                         

# Set the ticklabel font size globally
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22

N_try = conf.N_try
Fig_path = conf.Fig_path


@tf.function

# Plot results from TO and episode to check consistency
def plot_results(tau0,tau1,tau2,x_TO,y_TO,x_RL,y_RL,steps,to=0):

    timesteps = conf.dt*np.arange(steps+1)
    timesteps2 = conf.dt*np.arange(steps)
    fig = plt.figure(figsize=(12,8))
    if to:
        plt.suptitle('TO EXPLORATION: N try = {}'.format(N_try), y=1, fontsize=20)
    else:  
        plt.suptitle('POLICY EXPLORATION: N try = {}'.format(N_try), y=1, fontsize=20)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timesteps, x_TO, 'ro', linewidth=1, markersize=1) 
    ax1.plot(timesteps, y_TO, 'bo', linewidth=1, markersize=1)
    ax1.plot(timesteps, x_RL, 'go', linewidth=1, markersize=1) 
    ax1.plot(timesteps, y_RL, 'ko', linewidth=1, markersize=1)
    ax1.set_title('End-Effector Position',fontsize=20)
    ax1.legend(["x_TO","y_TO","x_RL","y_RL"],fontsize=20)
    ax1.set_xlabel('Time [s]',fontsize=20)
    ax1.set_ylabel('[m]',fontsize=20)    

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(timesteps2, tau0, 'ro', linewidth=1, markersize=1) 
    ax2.plot(timesteps2, tau1, 'bo', linewidth=1, markersize=1) 
    ax2.plot(timesteps2, tau2, 'go', linewidth=1, markersize=1) 
    ax2.legend(['tau0','tau1','tau2'],fontsize=20) 
    ax2.set_xlabel('Time [s]',fontsize=20)
    ax2.set_title('Controls',fontsize=20)

    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.plot(x_TO, y_TO, 'ro', linewidth=1, markersize=1)
    ax3.plot(x_RL, y_RL, 'bo', linewidth=1, markersize=1)
    ax3.legend(['TO','RL'],fontsize=20)
    ax3.plot([x_TO[0]],[y_TO[0]],'ko',markersize=5)
    ax3.add_artist(ell1)
    ax3.add_artist(ell2) 
    ax3.add_artist(ell3) 
    ax3.plot([conf.TARGET_STATE[0]],[conf.TARGET_STATE[1]],'bo',markersize=5)
    ax3.set_xlim(-41, 31)
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Plane',fontsize=20)
    ax3.set_xlabel('X [m]',fontsize=20)
    ax3.set_ylabel('Y [m]',fontsize=20)
    ax3.set_ylim(-35, 35)
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    fig.tight_layout()
    plt.show()

# Plot policy rollout from a single initial state as well as state and control trajectories
def plot_policy(tau0,tau1,tau2,x,y,steps,n_updates, diff_loc=0, PRETRAIN=0):

    timesteps = conf.dt*np.arange(steps)
    fig = plt.figure(figsize=(12,8))
    plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(N_try,n_updates), y=1, fontsize=20)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1) 
    ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1)
    ax1.set_title('End-Effector Position',fontsize=20)
    ax1.legend(["x","y"],fontsize=20)
    ax1.set_xlabel('Time [s]',fontsize=20)
    ax1.set_ylabel('[m]',fontsize=20)    

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(timesteps, tau0, 'ro', linewidth=1, markersize=1) 
    ax2.plot(timesteps, tau1, 'bo', linewidth=1, markersize=1) 
    ax2.plot(timesteps, tau2, 'go', linewidth=1, markersize=1) 
    ax2.legend(['tau0','tau1','tau2'],fontsize=20) 
    ax2.set_xlabel('Time [s]',fontsize=20)
    ax2.set_title('Controls',fontsize=20)

    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.plot(x, y, 'ro', linewidth=1, markersize=1) 
    ax3.add_artist(ell1)
    ax3.add_artist(ell2) 
    ax3.add_artist(ell3) 
    ax3.plot([conf.x_base],[3*conf.l],'ko',markersize=5)   
    ax3.plot([conf.TARGET_STATE[0]],[conf.TARGET_STATE[1]],'b*',markersize=10) 
    ax3.set_xlim([-41, 31])
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Plane',fontsize=20)
    ax3.set_xlabel('X [m]',fontsize=20)
    ax3.set_ylabel('Y [m]',fontsize=20)
    ax3.set_ylim(-35, 35)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    fig.tight_layout()
    if PRETRAIN:
        plt.savefig(Fig_path+'/PolicyEvaluation_Pretrain_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    else:    
        if diff_loc==0:
            plt.savefig(Fig_path+'/PolicyEvaluation_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
        else:
            plt.savefig(Fig_path+'/Actor/PolicyEvaluation_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    plt.clf()
    plt.close(fig)

# Plot only policy rollouts from multiple initial states
def plot_policy_eval(x_list,y_list,n_updates, diff_loc=0, PRETRAIN=0):

    fig = plt.figure(figsize=(12,8))
    plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(N_try,n_updates), y=1, fontsize=20)
    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])

    ax = fig.add_subplot(1, 1, 1)
    for idx in range(len(x_list)):
        if idx == 0:
            ax.plot(x_list[idx], y_list[idx], 'ro', linewidth=1, markersize=1)
        elif idx == 1:
            ax.plot(x_list[idx], y_list[idx], 'bo', linewidth=1, markersize=1)
        elif idx == 2:
            ax.plot(x_list[idx], y_list[idx], 'go', linewidth=1, markersize=1)
        elif idx == 3:
            ax.plot(x_list[idx], y_list[idx], 'co', linewidth=1, markersize=1)
        elif idx == 4:
            ax.plot(x_list[idx], y_list[idx], 'yo', linewidth=1, markersize=1)
        elif idx == 5:
            ax.plot(x_list[idx], y_list[idx], color='tab:orange', marker='o', markersize=1, linestyle='None')
        elif idx == 6:
            ax.plot(x_list[idx], y_list[idx], color='grey', marker='o', markersize=1, linestyle='None')
        elif idx == 7:
            ax.plot(x_list[idx], y_list[idx], color='tab:pink', marker='o', markersize=1, linestyle='None')
        elif idx == 8:
            ax.plot(x_list[idx], y_list[idx], color='lime', marker='o', markersize=1, linestyle='None')            
        ax.plot(x_list[idx][0],y_list[idx][0],'ko',markersize=5)
    ax.plot(conf.TARGET_STATE[0],conf.TARGET_STATE[1],'b*',markersize=10)
    ax.add_artist(ell1)
    ax.add_artist(ell2) 
    ax.add_artist(ell3)     
    ax.set_xlim([-41, 31])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]',fontsize=20)
    ax.set_ylabel('Y [m]',fontsize=20)
    ax.set_ylim(-35, 35)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    if PRETRAIN:
        plt.savefig(Fig_path+'/PolicyEvaluationMultiInit_Pretrain_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    else:    
        if diff_loc==0:
            plt.savefig(Fig_path+'/PolicyEvaluationMultiInit_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
        else:
            plt.savefig(Fig_path+'/Actor/PolicyEvaluationMultiInit_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    plt.clf()
    plt.close(fig)

# Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval()
def rolloutM(update_step_cntr, actor_model, env, rand_time, N_try, diff_loc=0, PRETRAIN=0):

    init_states_sim = [np.array([math.pi/4,-math.pi/8,-math.pi/8,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,math.pi/8,math.pi/8,0.0,0.0,0.0,0.0]),np.array([math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi,0.0,0.0,0.0,0.0,0.0,0.0])]
    tau0_all_sim,tau1_all_sim,tau2_all_sim = [],[],[]
    x_ee_all_sim, y_ee_all_sim = [], []

    for k in range(len(init_states_sim)):
        tau0_arr_sim,tau1_arr_sim,tau2_arr_sim = [],[],[]
        x_ee_arr_sim = [env.get_end_effector_position(init_states_sim[k])[0]]
        y_ee_arr_sim = [env.get_end_effector_position(init_states_sim[k])[1]]
        prev_state_sim = np.copy(init_states_sim[k])
        episodic_reward_sim = 0

        for i in range(conf.NSTEPS-1):
            if conf.NORMALIZE_INPUTS:
                prev_state_sim_norm = prev_state_sim / conf.state_norm_arr
                prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
            else:
                tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)
            ctrl = actor_model(tf_x_sim)
            ctrl_sim = tf.squeeze(ctrl).numpy()

            next_state_sim = env.simulate(prev_state_sim,ctrl_sim)
            rwrd_sim = env.reward(rand_time, next_state_sim,ctrl_sim)

            episodic_reward_sim += rwrd_sim
            tau0_arr_sim.append(ctrl_sim[0])
            tau1_arr_sim.append(ctrl_sim[1])
            tau2_arr_sim.append(ctrl_sim[2])
            x_ee_arr_sim.append(env.get_end_effector_position(next_state_sim)[0])
            y_ee_arr_sim.append(env.get_end_effector_position(next_state_sim)[1])
            prev_state_sim = np.copy(next_state_sim)

            if i==conf.NSTEPS-2:
                if conf.NORMALIZE_INPUTS:
                    prev_state_sim_norm = prev_state_sim / conf.state_norm_arr
                    prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
                else:
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)                    
                ctrl = actor_model(tf_x_sim)
                ctrl_sim = tf.squeeze(ctrl).numpy()
                tau0_arr_sim.append(ctrl_sim[0])
                tau1_arr_sim.append(ctrl_sim[1])
                tau2_arr_sim.append(ctrl_sim[2])
        if k==2:
            plot_policy(tau0_arr_sim,tau1_arr_sim,tau2_arr_sim,x_ee_arr_sim,y_ee_arr_sim,conf.NSTEPS,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)
            print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(N_try,update_step_cntr,episodic_reward_sim))

        tau0_all_sim.append(np.copy(tau0_arr_sim))            
        tau1_all_sim.append(np.copy(tau1_arr_sim))            
        tau2_all_sim.append(np.copy(tau2_arr_sim))            
        x_ee_all_sim.append(np.copy(x_ee_arr_sim))            
        y_ee_all_sim.append(np.copy(y_ee_arr_sim))

    plot_policy_eval(x_ee_all_sim,y_ee_all_sim,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)

    return  tau0_all_sim, tau1_all_sim, tau2_all_sim, x_ee_all_sim, y_ee_all_sim

def rolloutDI(update_step_cntr, actor_model, env, rand_time, N_try, diff_loc=0, PRETRAIN=0):

    init_states_sim = [np.array([math.pi/4,-math.pi/8,-math.pi/8,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,math.pi/8,math.pi/8,0.0,0.0,0.0,0.0]),np.array([math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi,0.0,0.0,0.0,0.0,0.0,0.0])]
    tau0_all_sim,tau1_all_sim = [], []
    x_ee_all_sim, y_ee_all_sim = [], []

    for k in range(len(init_states_sim)):
        tau0_arr_sim,tau1_arr_sim,tau2_arr_sim = [],[],[]
        x_ee_arr_sim = init_states_sim[k][0] #[conf.x_base + conf.l*(math.cos(init_states_sim[k][0]) + math.cos(init_states_sim[k][0]+init_states_sim[k][1]) + math.cos(init_states_sim[k][0]+init_states_sim[k][1]+init_states_sim[k][2]))]
        y_ee_arr_sim = init_states_sim[k][1] #[conf.y_base + conf.l*(math.sin(init_states_sim[k][0]) + math.sin(init_states_sim[k][0]+init_states_sim[k][1]) + math.sin(init_states_sim[k][0]+init_states_sim[k][1]+init_states_sim[k][2]))]
        prev_state_sim = np.copy(init_states_sim[k])
        episodic_reward_sim = 0

        for i in range(conf.NSTEPS-1):
            if conf.NORMALIZE_INPUTS:
                prev_state_sim_norm = prev_state_sim / conf.state_norm_arr
                prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
            else:
                tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)
            ctrl = actor_model(tf_x_sim)
            ctrl_sim = tf.squeeze(ctrl).numpy()

            next_state_sim = env.simulate(prev_state_sim,ctrl_sim)
            rwrd_sim = env.reward(rand_time, next_state_sim,ctrl_sim)

            episodic_reward_sim += rwrd_sim
            tau0_arr_sim.append(ctrl_sim[0])
            tau1_arr_sim.append(ctrl_sim[1])
            x_ee_arr_sim.append(next_state_sim[0])
            y_ee_arr_sim.append(next_state_sim[1])
            prev_state_sim = np.copy(next_state_sim)

            if i==conf.NSTEPS-2:
                if conf.NORMALIZE_INPUTS:
                    prev_state_sim_norm = prev_state_sim / conf.state_norm_arr
                    prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
                else:
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)                    
                ctrl = actor_model(tf_x_sim)
                ctrl_sim = tf.squeeze(ctrl).numpy()
                tau0_arr_sim.append(ctrl_sim[0])
                tau1_arr_sim.append(ctrl_sim[1])
        if k==2:
            plot_policy(tau0_arr_sim,tau1_arr_sim,x_ee_arr_sim,y_ee_arr_sim,conf.NSTEPS,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)
            print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(N_try,update_step_cntr,episodic_reward_sim))

        tau0_all_sim.append(np.copy(tau0_arr_sim))            
        tau1_all_sim.append(np.copy(tau1_arr_sim))            
        x_ee_all_sim.append(np.copy(x_ee_arr_sim))            
        y_ee_all_sim.append(np.copy(y_ee_arr_sim))

    plot_policy_eval(x_ee_all_sim,y_ee_all_sim,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)

    return  tau0_all_sim, tau1_all_sim, x_ee_all_sim, y_ee_all_sim

# Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized)
def plot_Return():
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(1, 1, 1)    
    ax.plot(ep_reward_list)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("N_try = {}".format(N_try))
    ax.grid(True)
    plt.savefig(Fig_path+'/EpReturn_Manipulator3DoF_{}'.format(N_try))
    plt.close()

# Plot average return considering 40 episodes 
def plot_AvgReturn():
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(avg_reward_list)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg. Return")
    ax.set_title("N_try = {}".format(N_try))
    ax.grid(True)
    plt.savefig(Fig_path+'/AvgReturn_Manipulator3DoF_{}'.format(N_try))
    plt.close()
