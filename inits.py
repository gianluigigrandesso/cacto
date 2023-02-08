''' Init functions to warm-start TO (Pyomo needs python functions to warm-start its variables)
'''
# Warmstarting with CACTO rollout
def init_tau(m,k,idx,init_TO_controls):
    return init_TO_controls[idx][k]

def init_x(m,k,idx,init_TO_states):
    return init_TO_states[idx][k]
    
# Warmstarting with initial conditions
def init_x_ICS(m,k,idx,prev_state):
    return prev_state[idx]

# Warmstarting with zeros
def init_0(m,k, idx=None):
    return 0.0