import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import array2tensor
from matplotlib import cm, colors
from matplotlib.patches import Ellipse

class PLOT():
    def __init__(self, N_try, env, conf):
        '''    
        :input N_try :                          (Test number)

        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)
            :param fig_ax_lim :                 (float array) Figure axis limit [x_min, x_max, y_min, y_max]
            :param Fig_path :                   (str) Figure path
            :param NSTEPS :                     (int) Max episode length
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param dt :                         (float) Timestep
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
        '''
        self.env = env    
        self.conf = conf

        self.N_try = N_try

        self.xlim = conf.fig_ax_lim[0].tolist()
        self.ylim = conf.fig_ax_lim[1].tolist()

        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22

        return 

    def plot_policy(self, tau, x, y, steps, n_updates, diff_loc=0, PRETRAIN=0):
        ''' Plot policy rollout from a single initial state as well as state and control trajectories '''
        timesteps = self.conf.dt*np.arange(steps)
        
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1, fontsize=20)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1) 
        ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1)
        ax1.set_title('End-Effector Position',fontsize=20)
        ax1.legend(["x","y"],fontsize=20)
        ax1.set_xlabel('Time [s]',fontsize=20)
        ax1.set_ylabel('[m]',fontsize=20)    

        col = ['ro', 'bo', 'go']
        legend_list = []
        ax2 = fig.add_subplot(2, 2, self.conf.nb_action)
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps, tau[:,i], col[i], linewidth=1, markersize=1) 
            legend_list.append('tau{}'.format(i))
        ax2.legend(legend_list,fontsize=20) 
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

        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=10) 
        ax3.set_xlim(self.xlim)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Plane',fontsize=20)
        ax3.set_xlabel('X [m]',fontsize=20)
        ax3.set_ylabel('Y [m]',fontsize=20)
        ax3.set_ylim(self.xlim)

        for ax in [ax1, ax2, ax3]:
            ax.grid(True)
        fig.tight_layout()
        if PRETRAIN:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluation_Pretrain_{}_{}'.format(self.N_try,n_updates))
        else:    
            if diff_loc==0:
                plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}'.format(self.N_try,n_updates))
            else:
                plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}'.format(self.N_try,n_updates))
        plt.clf()
        plt.close(fig)

    def plot_policy_eval(self,x_list,y_list,n_updates, diff_loc=0, PRETRAIN=0):
        ''' Plot only policy rollouts from multiple initial states '''
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1, fontsize=20)
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
        ax.set_xlim(self.xlim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]',fontsize=20)
        ax.set_ylabel('Y [m]',fontsize=20)
        ax.set_ylim(self.xlim)
        ax.grid(True)
        fig.tight_layout()
        if PRETRAIN:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_Pretrain_{}_{}'.format(self.N_try,n_updates))
        else:    
            if diff_loc==0:
                plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}'.format(self.N_try,n_updates))
            else:
                plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}'.format(self.N_try,n_updates))
        #plt.show()
        plt.clf()
        plt.close(fig)

    def rollout(self,update_step_cntr, actor_model, init_states_sim, diff_loc=0, PRETRAIN=0):
        ''' Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval() '''
        tau_all_sim = np.empty((len(init_states_sim)*(self.conf.NSTEPS),self.conf.nb_action))
        x_ee_all_sim = []
        y_ee_all_sim = []

        returns = {}

        for k in range(len(init_states_sim)):
            x_ee_arr_sim = [self.env.get_end_effector_position(init_states_sim[k])[0]]
            y_ee_arr_sim = [self.env.get_end_effector_position(init_states_sim[k])[1]]
            prev_state_sim = np.copy(init_states_sim[k])
            episodic_reward_sim = 0

            for i in range(self.conf.NSTEPS-1):
                if self.conf.NORMALIZE_INPUTS:
                    prev_state_sim_norm = prev_state_sim / self.conf.state_norm_arr
                    prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                    tf_x_sim = array2tensor(prev_state_sim_norm)
                else:
                    tf_x_sim = array2tensor(prev_state_sim)
                ctrl = actor_model(tf_x_sim)
                ctrl_sim = tf.squeeze(ctrl).numpy()

                next_state_sim, rwrd_sim = self.env.step(self.conf.cost_weights_running, prev_state_sim,ctrl_sim)

                episodic_reward_sim += rwrd_sim

                tau_all_sim[i + k*(self.conf.NSTEPS), :] = ctrl_sim
                x_ee_arr_sim.append(self.env.get_end_effector_position(next_state_sim)[0])
                y_ee_arr_sim.append(self.env.get_end_effector_position(next_state_sim)[1])
                prev_state_sim = np.copy(next_state_sim)

                if i==self.conf.NSTEPS-2:
                    if self.conf.NORMALIZE_INPUTS:
                        prev_state_sim_norm = prev_state_sim / self.conf.state_norm_arr
                        prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                        tf_x_sim = array2tensor(prev_state_sim_norm)
                    else:
                        tf_x_sim = array2tensor(prev_state_sim)                   
                    ctrl = actor_model(tf_x_sim)
                    ctrl_sim = tf.squeeze(ctrl).numpy()
                    tau_all_sim[i + k*(self.conf.NSTEPS), :] = ctrl_sim
            if k==2:
                #self.plot_policy(tau_all_sim[(k-1)*(self.conf.NSTEPS):k*(self.conf.NSTEPS)], x_ee_arr_sim, y_ee_arr_sim, self.conf.NSTEPS, update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)
                print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(self.N_try,update_step_cntr,episodic_reward_sim))
                
            x_ee_all_sim.append(np.copy(x_ee_arr_sim))  
            y_ee_all_sim.append(np.copy(y_ee_arr_sim))

            returns[init_states_sim[k][0],init_states_sim[k][1]] = episodic_reward_sim

        self.plot_policy_eval(x_ee_all_sim,y_ee_all_sim,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)

        return returns

    def plot_results(self, tau, ee_pos_TO, ee_pos_RL, steps, to=0):
        ''' Plot results from TO and episode to check consistency '''
        timesteps = self.conf.dt*np.arange(steps+1)
        fig = plt.figure(figsize=(12,8))
        if to:
            plt.suptitle('TO EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)
        else:  
            plt.suptitle('POLICY EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, ee_pos_TO[:,0], 'ro', linewidth=1, markersize=1) 
        ax1.plot(timesteps, ee_pos_TO[:,1], 'bo', linewidth=1, markersize=1)
        ax1.plot(timesteps, ee_pos_RL[:,0], 'go', linewidth=1, markersize=1) 
        ax1.plot(timesteps, ee_pos_RL[:,1], 'ko', linewidth=1, markersize=1)
        ax1.set_title('End-Effector Position',fontsize=20)
        ax1.legend(["x_TO","y_TO","x_RL","y_RL"],fontsize=20)
        ax1.set_xlabel('Time [s]',fontsize=20)
        ax1.set_ylabel('[m]',fontsize=20)    
        ax1.set_xlim(0, timesteps[-1])

        ax2 = fig.add_subplot(2, 2, 3)
        col = ['ro', 'bo', 'go']
        legend_list = []
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps[:-1], tau[:,i], col[i], linewidth=1, markersize=1) 
            legend_list.append('tau{}'.format(i))
        ax2.legend(legend_list,fontsize=20) 
        ax2.set_xlabel('Time [s]',fontsize=20)
        ax2.set_title('Controls',fontsize=20)

        ell1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0)
        ell1.set_facecolor([30/255, 130/255, 76/255, 1])
        ell2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0)
        ell2.set_facecolor([30/255, 130/255, 76/255, 1])
        ell3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0)
        ell3.set_facecolor([30/255, 130/255, 76/255, 1])
        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(ee_pos_TO[:,0], ee_pos_TO[:,1], 'ro', linewidth=1, markersize=2)
        ax3.plot(ee_pos_RL[:,0], ee_pos_RL[:,1], 'bo', linewidth=1, markersize=2)
        ax3.legend(['TO','RL'],fontsize=20)
        ax3.plot([ee_pos_TO[0,0]],[ee_pos_TO[0,1]],'ro',markersize=5)
        ax3.plot([ee_pos_RL[0,0]],[ee_pos_RL[0,1]],'bo',markersize=5)
        ax3.add_artist(ell1)
        ax3.add_artist(ell2) 
        ax3.add_artist(ell3) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'go',markersize=5)
        ax3.set_xlim(self.xlim)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Plane',fontsize=20)
        ax3.set_xlabel('X [m]',fontsize=20)
        ax3.set_ylabel('Y [m]',fontsize=20)
        ax3.set_ylim(self.xlim)

        for ax in [ax1, ax2, ax3]:
            ax.grid(True)

        fig.tight_layout()
        #plt.show()

    def plot_Return(self, ep_reward_list):
        ''' Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized) '''
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)   
        ax.set_yscale('log') 
        ax.plot(ep_reward_list**2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("N_try = {}".format(self.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/EpReturn_{}'.format(self.N_try))
        plt.close()

    def plot_AvgReturn(self, avg_reward_list):
        ''' Plot average return considering 40 episodes '''
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')
        ax.plot(avg_reward_list**2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg. Return")
        ax.set_title("N_try = {}".format(self.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/AvgReturn_{}'.format(self.N_try))
        plt.close()

    def plot_Reward_function(self):
        ''' Plot reward function (only system_id = 'double_integrator')'''
        x = np.arange(-15, 15, 0.25)
        y = np.arange(-15, 15, 0.25)
        X, Y = np.meshgrid(x, y)
        plot_data = np.zeros((x.shape[0],y.shape[0]))*np.nan
        for k_y in range(int(y.shape[0])):
            for k_x in range(int(x.shape[0])):
                x_p = x[k_x]
                y_p = y[k_y]
                plot_data[k_y,k_x] = self.env.reward(np.array([x_p,y_p, 0.0, 0.0, 0.0]))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, plot_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.title('Reference')
        plt.savefig('Reference.png')

    def plot_Critic_Value_function(self, critic_model, n_update):
        ''' Plot Value function as learned by the critic (only system_id = 'double_integrator')'''
        x = np.arange(-5, 10, 0.25)
        y = np.arange(-5, 10, 0.25)
        X, Y = np.meshgrid(x, y)
        criticplot_data = np.zeros((x.shape[0],y.shape[0]))*np.nan
        ref_data = np.zeros((x.shape[0],y.shape[0]))*np.nan
        for k_y in range(int(y.shape[0])):
            for k_x in range(int(x.shape[0])):
                x_p = x[k_x]
                y_p = y[k_y]
                input_np = np.array([[x_p,y_p, 0.0, 0.0, 0.0]])
                input_tf =  tf.convert_to_tensor(input_np, dtype=tf.float32)/15
                criticplot_data[k_y,k_x] = critic_model(input_tf,training=False)
                ref_data[k_y,k_x] = self.env.reward(np.array([x_p,y_p, 0.0, 0.0, 0.0]))
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')

        #fig, ax = plt.subplots(figsize=(12,6),subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, -criticplot_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.plot_surface(X, Y, -ref_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #all_xy, reward_all_xy = buffer.sample_all()
        #for i in range(len(all_xy)):
        #    ax.scatter(all_xy[i,0], all_xy[i,1], reward_all_xy[i],color='b')
        plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
        #plt.savefig('N_try_{}__n_update_{}.png'.format(N_try, n_update))
        plt.show()
        plt.close()
