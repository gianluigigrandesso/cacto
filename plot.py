import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from pyomo.environ import *
from pyomo.dae import *

class PLOT():
    def __init__(self, env, conf):

        self.env = env    
        self.conf = conf
                
        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22

        return 


    @tf.function

    # Plot policy rollout from a single initial state as well as state and control trajectories
    def plot_policy(self, tau, x, y, steps, n_updates, diff_loc=0, PRETRAIN=0):

        timesteps = self.conf.dt*np.arange(steps)
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.conf.N_try,n_updates), y=1, fontsize=20)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1) 
        ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1)
        ax1.set_title('End-Effector Position',fontsize=20)
        ax1.legend(["x","y"],fontsize=20)
        ax1.set_xlabel('Time [s]',fontsize=20)
        ax1.set_ylabel('[m]',fontsize=20)    

        col = ['ro', 'bo', 'go']
        ax2 = fig.add_subplot(2, 2, self.conf.robot.na)
        for i in range(self.conf.robot.na):
            ax2.plot(timesteps, tau[:,i], col[i], linewidth=1, markersize=1) 
        #ax2.legend(['tau0','tau1','tau2'],fontsize=20) 
        ax2.set_xlabel('Time [s]',fontsize=20)
        ax2.set_title('Controls',fontsize=20)

        ell1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0)
        ell1.set_facecolor([30/255, 130/255, 76/255, 1])
        ell2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0)
        ell2.set_facecolor([30/255, 130/255, 76/255, 1])
        ell3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0)
        ell3.set_facecolor([30/255, 130/255, 76/255, 1])

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(x, y, 'ro', linewidth=1, markersize=1) 
        ax3.add_artist(ell1)
        ax3.add_artist(ell2) 
        ax3.add_artist(ell3) 

        #ax3.plot([conf.x_base],[3*conf.l],'ko',markersize=5)   

        ax3.plot([self.TARGET_STATE[0]],[self.TARGET_STATE[1]],'b*',markersize=10) 
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
            plt.savefig(self.Fig_path+'/PolicyEvaluation_Pretrain_Manipulator3DoF_3OBS_{}_{}'.format(self.conf.N_try,n_updates))
        else:    
            if diff_loc==0:
                plt.savefig(self.Fig_path+'/PolicyEvaluation_Manipulator3DoF_3OBS_{}_{}'.format(self.conf.N_try,n_updates))
            else:
                plt.savefig(self.Fig_path+'/Actor/PolicyEvaluation_Manipulator3DoF_3OBS_{}_{}'.format(self.conf.N_try,n_updates))
        plt.clf()
        plt.close(fig)

    # Plot only policy rollouts from multiple initial states
    def plot_policy_eval(self,x_list,y_list,n_updates, diff_loc=0, PRETRAIN=0):

        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.conf.N_try,n_updates), y=1, fontsize=20)
        ell1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0)
        ell1.set_facecolor([30/255, 130/255, 76/255, 1])
        ell2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0)
        ell2.set_facecolor([30/255, 130/255, 76/255, 1])
        ell3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0)
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
        ax.plot(self.conf.TARGET_STATE[0],self.conf.TARGET_STATE[1],'b*',markersize=10)
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
            plt.savefig(self.conf.Fig_path+'/PolicyEvaluationMultiInit_Pretrain_Manipulator3DoF_3OBS_{}_{}'.format(self.conf.N_try,n_updates))
        else:    
            if diff_loc==0:
                plt.savefig(self.conf.Fig_path+'/PolicyEvaluationMultiInit_Manipulator3DoF_3OBS_{}_{}'.format(self.conf.N_try,n_updates))
            else:
                plt.savefig(self.conf.Fig_path+'/Actor/PolicyEvaluationMultiInit_Manipulator3DoF_3OBS_{}_{}'.format(self.conf.N_try,n_updates))
        plt.clf()
        plt.close(fig)

    # Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval()
    def rollout(self,update_step_cntr, actor_model, env, rand_time, init_states_sim, diff_loc=0, PRETRAIN=0):

        #init_states_sim = [np.array([math.pi/4,-math.pi/8,-math.pi/8,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,math.pi/8,math.pi/8,0.0,0.0,0.0,0.0]),np.array([math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi,0.0,0.0,0.0,0.0,0.0,0.0])]
        tau_all_sim = np.empty((len(init_states_sim)*(self.conf.NSTEPS),self.conf.robot.na))
        x_ee_arr_sim = np.empty(self.conf.NSTEPS)
        y_ee_arr_sim = np.empty(self.conf.NSTEPS)
        x_ee_all_sim = []
        y_ee_all_sim = []
        import time
        for k in range(len(init_states_sim)):
            #tau_arr_sim = []
            x_ee_arr_sim[0] = env.get_end_effector_position(init_states_sim[k])[0]
            y_ee_arr_sim[0] = env.get_end_effector_position(init_states_sim[k])[1]
            prev_state_sim = np.copy(init_states_sim[k])
            episodic_reward_sim = 0

            for i in range(self.conf.NSTEPS-1):
                if self.conf.NORMALIZE_INPUTS:
                    prev_state_sim_norm = prev_state_sim / self.conf.state_norm_arr
                    prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
                else:
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)
                ctrl = actor_model(tf_x_sim)
                ctrl_sim = tf.squeeze(ctrl).numpy()

                next_state_sim = env.simulate(prev_state_sim,ctrl_sim)
                rwrd_sim = env.reward(rand_time, next_state_sim,ctrl_sim)

                episodic_reward_sim += rwrd_sim
                #tau_arr_sim.append(ctrl_sim)
                tau_all_sim[i + k*(self.conf.NSTEPS), :] = ctrl_sim
                x_ee_arr_sim[i + 1] = env.get_end_effector_position(next_state_sim)[0]
                y_ee_arr_sim[i + 1] = env.get_end_effector_position(next_state_sim)[1]
                prev_state_sim = np.copy(next_state_sim)

                if i==self.conf.NSTEPS-2:
                    if self.conf.NORMALIZE_INPUTS:
                        prev_state_sim_norm = prev_state_sim / self.conf.state_norm_arr
                        prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                        tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
                    else:
                        tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)                    
                    ctrl = actor_model(tf_x_sim)
                    ctrl_sim = tf.squeeze(ctrl).numpy()
                    tau_all_sim[i + k*(self.conf.NSTEPS), :] = ctrl_sim
                    #tau_arr_sim.append(ctrl_sim)
            if k==2:
                #self.plot_policy(tau_all_sim[(self.conf.NSTEPS):2*(self.conf.NSTEPS)],x_ee_arr_sim,y_ee_arr_sim,self.conf.NSTEPS,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)
                print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(self.conf.N_try,update_step_cntr,episodic_reward_sim))

            #tau_all_sim.append(np.copy(tau_arr_sim))             
            x_ee_all_sim.append(np.copy(x_ee_arr_sim))  
            y_ee_all_sim.append(np.copy(y_ee_arr_sim))

        self.plot_policy_eval(x_ee_all_sim,y_ee_all_sim,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)

        return  tau_all_sim, x_ee_all_sim, y_ee_all_sim

    # Plot results from TO and episode to check consistency
    def plot_results(self,tau, x_TO,y_TO,x_RL,y_RL,steps,to=0):

        timesteps = self.conf.dt*np.arange(steps+1)
        timesteps2 = self.conf.dt*np.arange(steps)
        fig = plt.figure(figsize=(12,8))
        if to:
            plt.suptitle('TO EXPLORATION: N try = {}'.format(self.confN_try), y=1, fontsize=20)
        else:  
            plt.suptitle('POLICY EXPLORATION: N try = {}'.format(self.conf.N_try), y=1, fontsize=20)

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
        col = ['ro', 'bo', 'go']
        for i in range(self.conf.robot.na):
            ax2.plot(timesteps, tau[:,i], col[i], linewidth=1, markersize=1) 
        #ax2.legend(['tau0','tau1','tau2'],fontsize=20) 
        ax2.set_xlabel('Time [s]',fontsize=20)
        ax2.set_title('Controls',fontsize=20)

        ell1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0)
        ell1.set_facecolor([30/255, 130/255, 76/255, 1])
        ell2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0)
        ell2.set_facecolor([30/255, 130/255, 76/255, 1])
        ell3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0)
        ell3.set_facecolor([30/255, 130/255, 76/255, 1])
        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(x_TO, y_TO, 'ro', linewidth=1, markersize=1)
        ax3.plot(x_RL, y_RL, 'bo', linewidth=1, markersize=1)
        ax3.legend(['TO','RL'],fontsize=20)
        ax3.plot([x_TO[0]],[y_TO[0]],'ko',markersize=5)
        ax3.add_artist(ell1)
        ax3.add_artist(ell2) 
        ax3.add_artist(ell3) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'bo',markersize=5)
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

    # Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized)
    def plot_Return(self, ep_reward_list):
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)    
        ax.plot(ep_reward_list)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("N_try = {}".format(self.conf.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/EpReturn_Manipulator3DoF_{}'.format(self.conf.N_try))
        plt.close()

    # Plot average return considering 40 episodes 
    def plot_AvgReturn(self, avg_reward_list):
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(avg_reward_list)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg. Return")
        ax.set_title("N_try = {}".format(self.conf.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/AvgReturn_Manipulator3DoF_{}'.format(self.conf.N_try))
        plt.close()