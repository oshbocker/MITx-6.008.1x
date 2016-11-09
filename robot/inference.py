#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot
from pprint import pprint


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    num_hidden_states = len(all_possible_hidden_states)
    
    # TODO: Compute the forward messages
    forward_messages = np.zeros((num_time_steps,num_hidden_states))
    for i in range(num_hidden_states):    
        forward_messages[0,i] = prior_distribution[all_possible_hidden_states[i]]
        
    # Create a dictionary to index observed states
    observation_index_dict = {all_possible_observed_states[i]:i for i in range(len(all_possible_observed_states))}
    
    # Create a transition dictionary based on all possible hidden states
    transition_dict = {}
    for i in all_possible_hidden_states:
        transition_dict[i] = {}
        transitions = transition_model(i)
        for j in all_possible_hidden_states:
            if j in transitions:
                transition_dict[i][j] = transitions[j]
            else:
                transition_dict[i][j] = 0
    
    # Convert transition dictionary to a numpy array
    A = np.array([[transition_dict[i][j] for j in all_possible_hidden_states] for i in all_possible_hidden_states])        
    
    # Create an emission dictionary based on all possible hidden states
    emission_dict = {}
    for i in all_possible_hidden_states:
        emission_dict[i] = {}
        obs = observation_model(i)
        for j in all_possible_observed_states:
            if j in obs:
                emission_dict[i][j] = obs[j]
            else:
                emission_dict[i][j] = 0
    
    # Convert emission dictionary to a numpy array
    B = np.array([[emission_dict[i][j] for j in all_possible_observed_states] for i in all_possible_hidden_states])
    
    # Iterate through observations and calculate forward messages
    for o in range(num_time_steps-1):
        if observations[o] == None:
            forward_messages[o+1,:] = (forward_messages[o]) @ A
        else:
            obs = observation_index_dict[observations[o]]
            forward_messages[o+1,:] = (B[:,obs]*forward_messages[o]) @ A

    # TODO: Compute the backward messages      
    backward_messages = np.zeros((num_time_steps,len(all_possible_hidden_states)))
    backward_messages[num_time_steps-1,:] = 1/num_hidden_states
    
    # Iterate through observations in reverse order and calculate backwards messages
    for o in reversed(range(1,num_time_steps)):
        if observations[o] == None:
            backward_messages[o-1,:] = (backward_messages[o]) @ A.T
        else:
            obs = observation_index_dict[observations[o]]
            backward_messages[o-1,:] = (B[:,obs]*backward_messages[o]) @ A.T
            
    # TODO: Compute the marginals 
    marginals = [None] * num_time_steps # remove this
    marginal_matrix = np.zeros((num_time_steps,num_hidden_states))
    
    for o in range(num_time_steps):
        if observations[o] == None:
            marginal = forward_messages[o,:]*backward_messages[o,:]
        else:
            obs = observation_index_dict[observations[o]]
            marginal = forward_messages[o,:]*backward_messages[o,:]*B[:,obs]
        marginal_dist = robot.Distribution()
        for m in range(num_hidden_states):
            marginal_dist[all_possible_hidden_states[m]] = marginal[m]
        marginal_dist.renormalize()
        marginals[o] = marginal_dist
    
    return marginals

def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of estimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)
    num_hidden_states = len(all_possible_hidden_states)
 
    messages = np.zeros((num_time_steps,num_hidden_states))
    for i in range(num_hidden_states):    
        messages[0,i] = -np.log2(prior_distribution[all_possible_hidden_states[i]])
    tracebacks = np.zeros((num_time_steps-1,num_hidden_states))
        
    # Create a dictionary to index observed states
    observation_index_dict = {all_possible_observed_states[i]:i for i in range(len(all_possible_observed_states))}
    
    # Create a transition dictionary based on all possible hidden states
    transition_dict = {}
    for i in all_possible_hidden_states:
        transition_dict[i] = {}
        transitions = transition_model(i)
        for j in all_possible_hidden_states:
            if j in transitions:
                transition_dict[i][j] = -np.log2(transitions[j])
            else:
                transition_dict[i][j] = -np.log2(0)
    
    # Convert transition dictionary to a numpy array
    psi = np.array([[transition_dict[i][j] for j in all_possible_hidden_states] for i in all_possible_hidden_states])   
    
    # Create an emission dictionary based on all possible hidden states
    observation_dict = {}
    for i in all_possible_hidden_states:
        observation_dict[i] = {}
        obs = observation_model(i)
        for j in all_possible_observed_states:
            if j in obs:
                observation_dict[i][j] = -np.log2(obs[j])
            else:
                observation_dict[i][j] = -np.log2(0)
    
    # Convert emission dictionary to a numpy array
    phi = np.array([[observation_dict[i][j] for j in all_possible_observed_states] for i in all_possible_hidden_states])

    # Iterate through observations and calculate forward messages
    for o in range(1,num_time_steps):
        if observations[o-1] == None:
            new_matrix = psi.T + messages[o-1,:]
        else:
            obs = observation_index_dict[observations[o-1]]
            new_matrix = phi[:,obs] + psi.T + messages[o-1,:]
        messages[o,:] = np.amin(new_matrix,axis=1)
        tracebacks[o-1,:] = np.argmin(new_matrix,axis=1)
    
    breadcrumbs = np.zeros(num_time_steps)
    breadcrumbs[-1] = np.argmin(phi[:,observation_index_dict[observations[-1]]] + messages[-1,:])
    for i in reversed(range(0,num_time_steps-1)):
        breadcrumbs[i] = tracebacks[i,int(breadcrumbs[i+1])]
   
    estimated_hidden_states = [all_possible_hidden_states[int(i)] for i in breadcrumbs]    
    return estimated_hidden_states[0:]


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    
    Refer to this paper and the Serial LVA algorithm:
    http://www2.ensc.sfu.ca/people/faculty/cavers/ENSC805/readings/42comm02-seshadri.pdf
    """

    # Set up counters for later use
    num_time_steps = len(observations)
    num_hidden_states = len(all_possible_hidden_states)
 
    # Set up the best and second best messages with initial probabilities
    best_messages = np.zeros((num_time_steps,num_hidden_states))
    for i in range(num_hidden_states):    
        best_messages[0,i] = -np.log2(prior_distribution[all_possible_hidden_states[i]])
    
    second_messages = np.zeros((num_time_steps,num_hidden_states))
    for i in range(num_hidden_states):    
        second_messages[0,i] = -np.log2(prior_distribution[all_possible_hidden_states[i]])
    
    best_tracebacks = np.zeros((num_time_steps-1,num_hidden_states))
    second_tracebacks = np.zeros((num_time_steps-1,num_hidden_states))      
    # Create a dictionary to index observed states
    observation_index_dict = {all_possible_observed_states[i]:i for i in range(len(all_possible_observed_states))}
    
    # Create a transition dictionary based on all possible hidden states
    transition_dict = {}
    for i in all_possible_hidden_states:
        transition_dict[i] = {}
        transitions = transition_model(i)
        for j in all_possible_hidden_states:
            if j in transitions:
                transition_dict[i][j] = -np.log2(transitions[j])
            else:
                transition_dict[i][j] = -np.log2(0)
    
    # Convert transition dictionary to a numpy array
    psi = np.array([[transition_dict[i][j] for j in all_possible_hidden_states] for i in all_possible_hidden_states])   
    
    # Create an emission dictionary based on all possible hidden states
    observation_dict = {}
    for i in all_possible_hidden_states:
        observation_dict[i] = {}
        obs = observation_model(i)
        for j in all_possible_observed_states:
            if j in obs:
                observation_dict[i][j] = -np.log2(obs[j])
            else:
                observation_dict[i][j] = -np.log2(0)
    
    # Convert emission dictionary to a numpy array
    phi = np.array([[observation_dict[i][j] for j in all_possible_observed_states] for i in all_possible_hidden_states])

    # Define a function to get the second min
    def second_min(m, axis=1):
        if axis == None:
            sec = np.argsort(m)[1]
            sec_min = m[sec]
        else:
            sec = np.argsort(m, axis=axis)[:,1]
            sec_min = np.zeros(len(sec))
            for i in range(len(sec_min)):
                sec_min[i] = m[i,sec[i]]
        return sec_min

    # Initialize a merge array that indicates the second best path
    # has merged into the best path at any given state
    merge_array = np.zeros((num_time_steps,num_hidden_states))
    
    # Initialize the best and second best messages  
    if observations[0] == None:   
        init_matrix = psi.T + best_messages[0,:]
    else:
        obs = observation_index_dict[observations[0]]
        init_matrix = phi[:,obs] + psi.T + best_messages[0,:]       
    best_messages[1,:] = np.amin(init_matrix,axis=1)
    second_messages[1,:] = second_min(init_matrix)
    best_tracebacks[0,:] = np.argmin(init_matrix,axis=1)
    second_tracebacks[0,:] = np.argsort(init_matrix,axis=1)[:,1]
    merge_array[0,:] = False
    
    # Iterate through observations and calculate forward messages        
    for o in range(2,num_time_steps):
        if observations[o-1] == None:
            best_matrix = psi.T + best_messages[o-1,:]
            second_matrix = psi.T + second_messages[o-1,:]
        else:
            obs = observation_index_dict[observations[o-1]]
            best_matrix = phi[:,obs] + psi.T + best_messages[o-1,:]
            second_matrix = phi[:,obs] + psi.T + second_messages[o-1,:]
        best_messages[o,:] = np.amin(best_matrix,axis=1)
        best_tracebacks[o-1,:] = np.argmin(best_matrix,axis=1)
        option_1 = second_min(best_matrix)
        option_2 = np.amin(second_matrix,axis=1)
        # Compare each state in the best path and second best path
        for i in range(len(option_1)):
            if option_1[i] < option_2[i]:
                option_1_trace = np.argsort(best_matrix,axis=1)[:,1]
                second_messages[o,i] = option_1[i]
                second_tracebacks[o-1,i] = option_1_trace[i]
                merge_array[o-1,i] = True
            else:
                option_2_trace = np.argmin(second_matrix,axis=1)
                second_messages[o,i] = option_2[i]
                second_tracebacks[o-1,i] = option_2_trace[i]
                merge_array[o-1,i] = False
    
    # Set up breadcrumbs to follow tracebacks and get second best path
    breadcrumbs = np.zeros(num_time_steps)
    # Find out whether the last state is on the second bets path or best path
    option_1 = second_min(phi[:,observation_index_dict[observations[-1]]] + best_messages[-1,:],axis=None)
    option_2 = np.amin(phi[:,observation_index_dict[observations[-1]]] + second_messages[-1,:])
    # Need to understand once we have switched from second best path to best path    
    indicator = False
    # If we already switched from second best path to best than we just follow the best path
    if option_1 < option_2:
        breadcrumbs[-1] = np.argsort(phi[:,observation_index_dict[observations[-1]]] + best_messages[-1,:])[1]
        for i in reversed(range(0,num_time_steps-1)):
            breadcrumbs[i] = best_tracebacks[i,int(breadcrumbs[i+1])]
    else:
        # If we are on the second best path we look to the merge array to find out
        # when to merge with the best path        
        breadcrumbs[-1] = np.argmin(phi[:,observation_index_dict[observations[-1]]] + second_messages[-1,:])
        for i in reversed(range(0,num_time_steps-1)):
            if indicator == False:
                if merge_array[i,int(breadcrumbs[i+1])] == 1:
                    indicator = True
                # Until we get the merge indicator we stick with second best path
                breadcrumbs[i] = second_tracebacks[i,int(breadcrumbs[i+1])]
            else:
                # After merge we follow best path
                breadcrumbs[i] = best_tracebacks[i,int(breadcrumbs[i+1])]

    estimated_hidden_states = [all_possible_hidden_states[int(i)] for i in breadcrumbs] 

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
