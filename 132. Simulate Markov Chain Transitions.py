"""
Implement a function to simulate a Markov Chain. The function should take a 2D numpy array representing the transition matrix (where each row sums to 1),
an integer for the initial state index, and an integer for the number of steps to simulate. 
It should return a numpy array containing the sequence of state indices over time,
including the initial state. Use numpy for array operations and random selections.

Example:
Input:
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]]); print(simulate_markov_chain(transition_matrix, 0, 3))
Output:
[0 0 1 1]
Reasoning:
The solution simulates a Markov chain by starting with the initial state (0) and iteratively selecting the next state based on the probabilities in the transition matrix.
For the given input, this process generates the sequence [0, 0, 1, 1] over three steps,
where the first state is the initial one, and subsequent states are chosen such that from state 0,
it stays at 0, then transitions to 1, and remains at 1.

Insights: 

- A Markov Chain is a stochastic (random) process that moves between a finite set of states over time.
- The behavior of the system is fully described by a transition matrix:
 -- Each row corresponds to a current state.
 -- Each column corresponds to a possible next state.
 -- Each row sums to 1, representing a probability distribution.

-So, effectively, we're seeing the current state of the system and predicting and moving on the next step and recording what we're doing in the system. Doesn't it look like how we can make a robot move ?


"""

#SOLUTION: 

import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):

    #Finding the number of states
    num_states = transition_matrix.shape[0]

    #Starting the state sequence
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state

    #main simulation loop
    for t in range(1, num_steps + 1):
        current_state = states[t - 1]
        
        #sampling the next state
        states[t] = np.random.choice(
            num_states,
            p=transition_matrix[current_state]
        )

    return states
