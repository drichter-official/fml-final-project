# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 16:21:28 2022

@author: ztec1
"""
import numpy as np
state_rewards = [-5, 0, 0, 0, 0, 0, 5]
final_state = [True, False, False, False, False, False, True]
Q_values = [[0.0, 0.0], 
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]] # Q(s, a) matrix. [left, right].

def select_epsilon_greedy_action(epsilon, state):
  """Take random action with probability epsilon, else take best action."""
  result = np.random.uniform()
  if result < epsilon:
    return np.random.randint(0, 2) # Random action (left or right).
  else:
    return np.argmax(Q_values[state]) # Greedy action for state.

def apply_action(state, action):
  """Applies the selected action and get reward and next state.
  Action 0 means move to the left and action 1 means move to the right.
  """
  if action == 0:
    next_state = state-1
  else:
    next_state = state+1
  return state_rewards[next_state], next_state

num_episodes = 1000
epsilon = 0.2
discount = 0.9 # Change to 1.0 if you want to simplify Q-value results.

for episode in range(num_episodes+1):
  initial_state = 3 # State in the middle.
  state = initial_state
  while not final_state[state]: # Run until the end of the episode.
    # Select action.
    action = select_epsilon_greedy_action(epsilon, state)
    reward, next_state = apply_action(state, action)
    # Improve Q-values with Bellman Equation.
    if final_state[next_state]:
      Q_values[state][action] = reward
    else:
      Q_values[state][action] = reward + discount * max(Q_values[next_state])
    state = next_state

# Print Q-values to see if action right is always better than action left
# except for states 0 and 6, which are terminal states and you cannot take
# any action from them, so it does not matter.
print('Final Q-values are:')
print(Q_values)
action_dict = {0:'left', 1:'right'}
state = 0
for state, Q_vals in enumerate(Q_values):
  print('Best action for state {} is {}'.format(state, 
                                             action_dict[np.argmax(Q_vals)]))