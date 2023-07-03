import os
import math
import random
import gepetto#.corbaserver
import time
import subprocess
import numpy as np
import pinocchio as pin
from numpy.linalg import norm
from pinocchio.robot_wrapper import RobotWrapper as PinocchioRobotWrapper

class Contact:
    ''' A contact between a contact-point and a contact-surface
    '''
    def __init__(self, contact_point, contact_surface):
        self.cp = contact_point
        self.cs = contact_surface
        self.reset_contact_position()

    def reset_contact_position(self):
        # Initialize anchor point p0, that is the initial (0-load) position of the spring
        self.p0 = self.cp.get_position()
        self.in_contact = True

    def compute_force(self):
        self.f, self.p0 = self.cs.compute_force(self.cp, self.p0)
        return self.f
        
    def get_jacobian(self):
        return self.cp.get_jacobian()

class RobotWrapper(PinocchioRobotWrapper):
    
    @staticmethod
    def BuildFromURDF(filename, package_dirs=None, root_joint=None, verbose=False, meshLoader=None):
        robot = RobotWrapper()
        robot.initFromURDF(filename, package_dirs, root_joint, verbose, meshLoader)
        return robot
    
    @property
    def na(self):
        if(self.model.joints[0].nq==7):
            return self.model.nv-6
        return self.model.nv

    def mass(self, q, update=True):
        if(update):
            return pin.crba(self.model, self.data, q)
        return self.data.M

    def nle(self, q, v, update=True):
        if(update):
            return pin.nonLinearEffects(self.model, self.data, q, v)
        return self.data.nle
        
    def com(self, q=None, v=None, a=None, update=True):
        if(update==False or q is None):
            return PinocchioRobotWrapper.com(self, q)
        if a is None:
            if v is None:
                return PinocchioRobotWrapper.com(self, q)
            return PinocchioRobotWrapper.com(self, q, v)
        return PinocchioRobotWrapper.com(self, q, v,a)
        
    def Jcom(self, q, update=True):
        if(update):
            return pin.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom
        
    def momentumJacobian(self, q, v, update=True):
        if(update):
            pin.ccrba(self.model, self.data, q, v)
        return self.data.Ag

    def computeAllTerms(self, q, v):
        ''' pin.computeAllTerms is equivalent to calling:
            pinocchio::forwardKinematics
            pinocchio::crba
            pinocchio::nonLinearEffects
            pinocchio::computeJointJacobians
            pinocchio::centerOfMass
            pinocchio::jacobianCenterOfMass
            pinocchio::kineticEnergy
            pinocchio::potentialEnergy
            This is too much for our needs, so we call only the functions
            we need, including those for the frame kinematics
        '''
        #pin.computeAllTerms(self.model, self.data, q, v);
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)
        pin.nonLinearEffects(self.model, self.data, q, v)
       
    def forwardKinematics(self, q, v=None, a=None):
        if v is not None:
            if a is not None:
                pin.forwardKinematics(self.model, self.data, q, v, a)
            else:
                pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)

    def inverseKinematics(self, ee_coords):
        ''' Inverse kinematics algorithm to compute a joint configuration given the EE coordinates'''
        
        oMdes = pin.SE3(np.eye(3), np.array([ee_coords[0], ee_coords[1], 0.0]))
        random.seed(123)
        for j in range(100):
            q = np.array([random.uniform(-math.pi,math.pi),random.uniform(-math.pi,math.pi),random.uniform(-math.pi,math.pi)])
            # q = pin.neutral(self.model)

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            eps    = 1e-4
            IT_MAX = 1000
            DT     = 1e-1
            damp   = 1e-12
            frame_id = self.model.getFrameId('fixed_ee')    
            i=0
            while True:
                pin.forwardKinematics(self.model,self.data,q)
                pin.updateFramePlacements(self.model, self.data)
                dMi = oMdes.actInv(self.data.oMf[frame_id])
                err = pin.log(dMi).vector
                if norm(err) < eps:
                    success = True
                    break
                if i >= IT_MAX:
                    success = False
                    break
                J = pin.computeFrameJacobian(self.model,self.data,q,frame_id)
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model,q,v*DT)
                if not i % 10:
                    print('%d: error = %s' % (i, err.T))
                i += 1
            
            if success:
                print("Convergence achieved!")
                break
            else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision. Retry ({}) with another initial configuration".format(j))
                
        return q, success

    def frameJacobian(self, q, index, update=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        ''' Call computeFrameJacobian if update is true. If not, user should call computeFrameJacobian first.
            Then call getFrameJacobian and return the Jacobian matrix.
            ref_frame can be: ReferenceFrame.LOCAL, ReferenceFrame.WORLD, ReferenceFrame.LOCAL_WORLD_ALIGNED
        '''
        if(update): 
            pin.computeFrameJacobian(self.model, self.data, q, index)
        return pin.getFrameJacobian(self.model, self.data, index, ref_frame)
        
    def frameVelocity(self, q, v, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v)
        v_local = pin.getFrameVelocity(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return v_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            v_world = H.act(v_local)
            return v_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        v = Hr.act(v_local)
        return v
            
    def frameAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
        
    def frameClassicAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        v = pin.getFrameVelocity(self.model, self.data, index)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        a_local.linear += np.cross(v.angular, v.linear, axis=0)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
      
    def deactivateCollisionPairs(self, collision_pair_indexes):
        for i in collision_pair_indexes:
            self.collision_data.deactivateCollisionPair(i)
            
    def addAllCollisionPairs(self):
        self.collision_model.addAllCollisionPairs()
        self.collision_data = pin.GeometryData(self.collision_model)
        
    def isInCollision(self, q, stop_at_first_collision=True):
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, np.asmatrix(q).reshape((self.model.nq,1)), stop_at_first_collision)

    def findFirstCollisionPair(self, consider_only_active_collision_pairs=True):
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    return (i, self.collision_model.collisionPairs[i])
        return None
        
    def findAllCollisionPairs(self, consider_only_active_collision_pairs=True):
        res = []
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    res += [(i, self.collision_model.collisionPairs[i])]
        return res

class RobotSimulator:

    def __init__(self, robot, q0_init, v0_init, simulation_type, tau_coulomb_max, use_viewer=False, DISPLAY_T=None, CAMERA_TRANSFORM=None, show_floor=False):
        self.robot = robot
        self.model = self.robot.model
        self.data = self.model.createData()
        self.t = 0.0                            # time
        self.nv = nv = self.model.nv            # Dimension of joint velocities vector
        self.na = na = robot.na                 # number of actuated joints
        self.S = np.hstack((np.zeros((na, nv-na)), np.eye(na, na))) # Matrix S used as filter of vector of inputs U

        self.contacts = []
        self.candidate_contact_points = []      # candidate contact points
        self.contact_surfaces = []
        
        self.DISPLAY_T = DISPLAY_T              # refresh period for viewer
        self.use_viewer = use_viewer
        self.tau_coulomb_max = tau_coulomb_max
        self.display_counter = self.DISPLAY_T
        self.init(q0_init, v0_init, True)
        
        self.tau_c = np.zeros(na)   # Coulomb friction torque
        self.simulation_type = simulation_type
        self.set_coulomb_friction(tau_coulomb_max)

        # for gepetto viewer
        self.gui = None
        if(use_viewer):
            try:
                prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(prompt[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
            except:
                pass
            gepetto.corbaserver.Client()
            gepetto.Client()
            self.robot.initViewer(loadModel=True)
            self.gui = self.robot.viewer.gui
            if(show_floor):
                self.robot.viewer.gui.createSceneWithFloor('world')
                self.gui.setLightingMode('world/floor', 'OFF')
            self.robot.displayCollisions(False)
            self.robot.displayVisuals(True)
            self.robot.display(self.q)            
            try:  
                self.gui.setCameraTransform("python-pinocchio", CAMERA_TRANSFORM)
            except:
                self.gui.setCameraTransform(0, CAMERA_TRANSFORM)            

    # Re-initialize the simulator
    def init(self, q0=None, v0=None, reset_contact_positions=False):
        self.first_iter = True

        if q0 is not None:
            self.q = q0.copy()
            
        if(v0 is None):
            self.v = np.zeros(self.robot.nv)
        else:
            self.v = v0.copy()
        self.dv = np.zeros(self.robot.nv)
        self.resize_contact_data(reset_contact_positions)
          
    def resize_contact_data(self, reset_contact_positions=False):
        self.nc = len(self.contacts)                    # number of contacts
        self.nk = 3*self.nc                             # size of contact force vector
        self.f = np.zeros(self.nk)                      # contact forces
        self.Jc = np.zeros((self.nk, self.model.nv))    # contact Jacobian

        # reset contact position
        if(reset_contact_positions):
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            for c in self.contacts:
                c.reset_contact_position()

        self.compute_forces(compute_data=True)
        
    def set_coulomb_friction(self, tau_max):
        self.tau_coulomb_max = 1e-2*tau_max*self.model.effortLimit        
        self.simulate_coulomb_friction = (norm(self.tau_coulomb_max)!=0.0)
        
    def collision_detection(self):
        for s in self.contact_surfaces:                 # for each contact surface
            for cp in self.candidate_contact_points:    # for each candidate contact point
                p = cp.get_position()
                if(s.check_collision(p)):               # check whether the point is colliding with the surface
                    if(not cp.active):                  # if the contact was not already active
                        print("Collision detected between point", cp.frame_name, " at ", p)
                        cp.active = True
                        cp.contact = Contact(cp, s)
                        self.contacts += [cp.contact]
                        self.resize_contact_data()
                else:
                    if(cp.active):
                        print("Contact lost between point", cp.frame_name, " at ", p)
                        cp.active = False
                        self.contacts.remove(cp.contact)
                        self.resize_contact_data()

    def compute_forces(self, compute_data=True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data:
            pin.forwardKinematics(self.model, self.data, self.q, self.v)
            # Computes the placements of all the operational frames according to the current joint placement stored in data
            pin.updateFramePlacements(self.model, self.data)
            self.collision_detection()
            
        i = 0
        for c in self.contacts:
            self.f[i:i+3] = c.compute_force()
            self.Jc[i:i+3, :] = c.get_jacobian()
            i += 3
        return self.f

    def step(self, u, dt=None):
        if dt is None:
            dt = self.dt

        # (Forces are directly in the world frame, and aba wants them in the end effector frame)
        pin.computeAllTerms(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        M = self.data.M
        h = self.data.nle
        self.collision_detection()
        self.compute_forces(False)

        if(self.simulate_coulomb_friction and self.simulation_type=='timestepping'):
            # minimize kinetic energy using time stepping
            from quadprog import solve_qp
            '''
            Solve a strictly convex quadratic program
            
            Minimize     1/2 x^T G x - a^T x
            Subject to   C.T x >= b
            
            Input Parameters
            ----------
            G : array, shape=(n, n)
            a : array, shape=(n,)
            C : array, shape=(n, m) matrix defining the constraints
            b : array, shape=(m), default=None, vector defining the constraints
            meq : int, default=0
                the first meq constraints are treated as equality constraints,
                all further as inequality constraints
            '''
            # M (v' - v) = dt*S^T*(tau - tau_c) - dt*h + dt*J^T*f
            # M v' = M*v + dt*(S^T*tau - h + J^T*f) - dt*S^T*tau_c
            # M v' = b + B*tau_c
            # v' = Minv*(b + B*tau_c)
            b = M.dot(self.v) + dt*(self.S.T.dot(u) - h + self.Jc.T.dot(self.f))
            B = - dt*self.S.T
            # Minimize kinetic energy:
            # min v'.T * M * v'
            # min  (b+B*tau_c​).T*Minv*(b+B*tau_c​) 
            # min tau_c.T * B.T*Minv*B* tau_C + 2*b.T*Minv*B*tau_c
            Minv = np.linalg.inv(M)
            G = B.T.dot(Minv.dot(B))
            a = -b.T.dot(Minv.dot(B))
            C = np.vstack((np.eye(self.na), -np.eye(self.na)))
            c = np.concatenate((-self.tau_coulomb_max, -self.tau_coulomb_max))
            solution = solve_qp(G, a, C.T, c, 0)
            self.tau_c = solution[0]
            self.v = Minv.dot(b + B.dot(self.tau_c))
            self.q = pin.integrate(self.model, self.q, self.v*dt)
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h + self.Jc.T.dot(self.f))
        elif(self.simulation_type=='euler' or self.simulate_coulomb_friction==False):
            self.tau_c = self.tau_coulomb_max*np.sign(self.v[-self.na:])
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h + self.Jc.T.dot(self.f))
            # v_mean = np.copy(self.v) + 0.5*dt*self.dv
            self.q = pin.integrate(self.model, self.q, self.v*dt)
            # self.q += dt*v_mean        
            self.v += self.dv*dt
        else:
            print("[ERROR] Unknown simulation type:", self.simulation_type)

        self.t += dt
        return self.q, self.v, self.dv

    def reset(self):
        self.first_iter = True

    def simulate(self, state, u, dt=0.05, ndt=1):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
        self.q = state[:self.robot.nq]
        self.v = state[self.robot.nq:self.robot.nv+self.robot.nq]

        tau_c_avg = 0*self.tau_c
        for i in range(ndt):
            self.q, self.v, self.dv = self.step(u, dt/ndt)
            tau_c_avg += self.tau_c
        self.tau_c = tau_c_avg / ndt

        if (self.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0.0:
                self.robot.display(self.q)
                self.display_counter = self.DISPLAY_T

        return self.q, self.v, self.f
        
    def display(self, q):
        if(self.use_viewer):
            self.robot.display(q)


class RobotWrapper(PinocchioRobotWrapper):
    
    @staticmethod
    def BuildFromURDF(filename, package_dirs=None, root_joint=None, verbose=False, meshLoader=None):
        robot = RobotWrapper()
        robot.initFromURDF(filename, package_dirs, root_joint, verbose, meshLoader)
        return robot
    
    @property
    def na(self):
        if(self.model.joints[0].nq==7):
            return self.model.nv-6
        return self.model.nv

    def mass(self, q, update=True):
        if(update):
            return pin.crba(self.model, self.data, q)
        return self.data.M

    def nle(self, q, v, update=True):
        if(update):
            return pin.nonLinearEffects(self.model, self.data, q, v)
        return self.data.nle
        
    def com(self, q=None, v=None, a=None, update=True):
        if(update==False or q is None):
            return PinocchioRobotWrapper.com(self, q)
        if a is None:
            if v is None:
                return PinocchioRobotWrapper.com(self, q)
            return PinocchioRobotWrapper.com(self, q, v)
        return PinocchioRobotWrapper.com(self, q, v,a)
        
    def Jcom(self, q, update=True):
        if(update):
            return pin.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom
        
    def momentumJacobian(self, q, v, update=True):
        if(update):
            pin.ccrba(self.model, self.data, q, v)
        return self.data.Ag

    def computeAllTerms(self, q, v):
        ''' pin.computeAllTerms is equivalent to calling:
            pinocchio::forwardKinematics
            pinocchio::crba
            pinocchio::nonLinearEffects
            pinocchio::computeJointJacobians
            pinocchio::centerOfMass
            pinocchio::jacobianCenterOfMass
            pinocchio::kineticEnergy
            pinocchio::potentialEnergy
            This is too much for our needs, so we call only the functions
            we need, including those for the frame kinematics
        '''
        #pin.computeAllTerms(self.model, self.data, q, v);
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)
        pin.nonLinearEffects(self.model, self.data, q, v)
       
    def forwardKinematics(self, q, v=None, a=None):
        if v is not None:
            if a is not None:
                pin.forwardKinematics(self.model, self.data, q, v, a)
            else:
                pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)

    def inverseKinematics(self, ee_coords):
        ''' Inverse kinematics algorithm to compute a joint configuration given the EE coordinates'''
        
        oMdes = pin.SE3(np.eye(3), np.array([ee_coords[0], ee_coords[1], 0.0]))
        random.seed(123)
        for j in range(100):
            q = np.array([random.uniform(-math.pi,math.pi),random.uniform(-math.pi,math.pi),random.uniform(-math.pi,math.pi)])
            # q = pin.neutral(self.model)

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            eps    = 1e-4
            IT_MAX = 1000
            DT     = 1e-1
            damp   = 1e-12
            frame_id = self.model.getFrameId('fixed_ee')    
            i=0
            while True:
                pin.forwardKinematics(self.model,self.data,q)
                pin.updateFramePlacements(self.model, self.data)
                dMi = oMdes.actInv(self.data.oMf[frame_id])
                err = pin.log(dMi).vector
                if norm(err) < eps:
                    success = True
                    break
                if i >= IT_MAX:
                    success = False
                    break
                J = pin.computeFrameJacobian(self.model,self.data,q,frame_id)
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model,q,v*DT)
                if not i % 10:
                    print('%d: error = %s' % (i, err.T))
                i += 1
            
            if success:
                print("Convergence achieved!")
                break
            else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision. Retry ({}) with another initial configuration".format(j))
                
        return q, success

    def frameJacobian(self, q, index, update=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        ''' Call computeFrameJacobian if update is true. If not, user should call computeFrameJacobian first.
            Then call getFrameJacobian and return the Jacobian matrix.
            ref_frame can be: ReferenceFrame.LOCAL, ReferenceFrame.WORLD, ReferenceFrame.LOCAL_WORLD_ALIGNED
        '''
        if(update): 
            pin.computeFrameJacobian(self.model, self.data, q, index)
        return pin.getFrameJacobian(self.model, self.data, index, ref_frame)
        
    def frameVelocity(self, q, v, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v)
        v_local = pin.getFrameVelocity(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return v_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            v_world = H.act(v_local)
            return v_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        v = Hr.act(v_local)
        return v
            
    def frameAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
        
    def frameClassicAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        v = pin.getFrameVelocity(self.model, self.data, index)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        a_local.linear += np.cross(v.angular, v.linear, axis=0)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.pin(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
      
    def deactivateCollisionPairs(self, collision_pair_indexes):
        for i in collision_pair_indexes:
            self.collision_data.deactivateCollisionPair(i)
            
    def addAllCollisionPairs(self):
        self.collision_model.addAllCollisionPairs()
        self.collision_data = pin.GeometryData(self.collision_model)
        
    def isInCollision(self, q, stop_at_first_collision=True):
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, np.asmatrix(q).reshape((self.model.nq,1)), stop_at_first_collision)

    def findFirstCollisionPair(self, consider_only_active_collision_pairs=True):
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    return (i, self.collision_model.collisionPairs[i])
        return None
        
    def findAllCollisionPairs(self, consider_only_active_collision_pairs=True):
        res = []
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    res += [(i, self.collision_model.collisionPairs[i])]
        return res