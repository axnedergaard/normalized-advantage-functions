#TODO improved logging, include if env solved

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
import argparse

import naf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #silence TF compilation warnings

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--graph', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--environment', dest='environment', nargs='+', type=str, default='InvertedPendulum-v1')
parser.add_argument('--repeats', dest='repeats', type=int, default=1)
parser.add_argument('--episodes', dest='episodes', type=int, default=1000)
parser.add_argument('--max_episode_steps', dest='max_episode_steps', type=int, default=1000)
parser.add_argument('--train_steps', dest='train_steps', type=int, default=5)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, nargs='+', default=0.01)
parser.add_argument('--batch_normalize', dest='batch_normalize', type=bool, default=True)
parser.add_argument('--gamma', dest='gamma', type=float,nargs='+', default=0.99)
parser.add_argument('--tau', dest='tau', type=float,nargs='+', default=0.99)
parser.add_argument('--epsilon', dest='epsilon', type=float, nargs='+', default=0.1)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, nargs='+', default=32)
parser.add_argument('--hidden_n', dest='hidden_n', type=int,nargs='+', default=2)
parser.add_argument('--hidden_activation', dest='hidden_activation', nargs='+', default=tf.nn.relu)
parser.add_argument('--batch_size', dest='batch_size', type=int, nargs='+', default=128)
parser.add_argument('--memory_capacity', dest='memory_capacity', type=int, nargs='+', default=10000)
parser.add_argument('-v', action='count', default=0) 
parser.add_argument('--load', dest='load_path', type=str, default=None)
parser.add_argument('--output', dest='output_path', type=str, default=None)
parser.add_argument('--covariance', dest='covariance', type=str, nargs='+', default="original")
parser.add_argument('--solve_threshold', dest='solve_threshold', type=float, nargs='+', default=None) #threshold for having solved environment
args = parser.parse_args()

def fill_episodes(rewards, n, value):
  return rewards + [value]*n

#get legends for plot
def recursive_legend(keys, remaining_vals, vals):
  if remaining_vals == []:
    legend = ""
    for i,l in enumerate(vals):
      legend += str(keys[i]) + "=" + str(l) + ","
    legend = legend[:-1] #remove last comma
    return [legend]
  else:
    legend = []
    for l in remaining_vals[0]:
      legend += recursive_legend(keys, remaining_vals[1:], vals + [l])
    return legend

#run experiment for every combination of hyperparameters
def recursive_experiment(keys, remaining_vals, vals):
  if remaining_vals == []:
    return [experiment(dict(zip(keys,vals)))]
  else:
    rewards = []
    if type(remaining_vals[0]) != list:
      rewards += recursive_experiment(keys, remaining_vals[1:], vals + [remaining_vals[0]])
    else: 
      for r in remaining_vals[0]:
        rewards += recursive_experiment(keys, remaining_vals[1:],  vals + [r])
    return rewards

def experiment(args):
  if args['v'] > 0:
    print("Experiment " + str(args)) 

  env = gym.make(args['environment'])
  
  experiments_rewards = []
  for i in range(args['repeats']):
    agent = naf.Agent(args['v'], env.observation_space, env.action_space, args['learning_rate'], args['batch_normalize'], args['gamma'], args['tau'], args['epsilon'], args['hidden_size'], args['hidden_n'], args['hidden_activation'], args['batch_size'], args['memory_capacity'], args['load_path'], args['covariance'])
    experiment_rewards = []
    terminate = None
    solved = 0 #only relevant if solved_threshold is set

    for j in range(args['episodes']):
      if terminate is not None:
        fill_value = 0
        if terminate == "solved":
          fill_value = args['solve_threshold']
        experiment_rewards = fill_episodes(experiment_rewards, args['episodes']-j, fill_value)
        break

      rewards = 0
      state = env.reset()
 
      for k in range(args['max_episode_steps']):
        if args['render']:
          env.render()
        
        action = agent.get_action(state)
        if np.isnan(np.min(action)): #if NaN action (neural network exploded)
          print("Warning: NaN action, terminating agent")
          with open("error.txt", "a") as error_file:
            error_file.write(str(args) + " repeat " + str(i) + " episode " + str(j) + " step " + str(k) + " NaN\n")
          rewards = 0 #TODO ?
          terminate = "nan"
          break
        #print(action)
        state_next,reward,terminal,_ = env.step(agent.scale(action, env.action_space.low, env.action_space.high))
        
        if k-1 >= args['max_episode_steps']:
          terminal = True
          
        agent.observe(state,action,reward,state_next,terminal)

        for l in range(args['train_steps']):
          agent.learn()

        state = state_next
        rewards += reward
        if terminal:
          agent.reset()
          break
      experiment_rewards += [rewards]

      if args['solve_threshold'] is not None:
        if rewards >= args['solve_threshold']:
          solved += 1
        else:
          solved = 0
        if solved >= 10: #number of repeated rewards above threshold to consider environment solved = 10
          print("[Solved]")
          terminate = "solved"

      if args['v'] > 0:
        print("Reward(" + str(i) + "," + str(j) + "," + str(k) + ")=" + str(rewards))
    if args['v'] > 1:
      print(np.mean(experiment_rewards[-10:]))
    experiments_rewards += [experiment_rewards]

  return experiments_rewards


#main

#construct experiment inputs
keys=['v', 'graph','render','environment','repeats','episodes','max_episode_steps','train_steps','batch_normalize', 'learning_rate','gamma','tau','epsilon','hidden_size','hidden_n','hidden_activation','batch_size', 'memory_capacity', 'load_path', 'covariance', 'solve_threshold']
vals=[args.v, args.graph, args.render, args.environment, args.repeats, args.episodes, args.max_episode_steps, args.train_steps, args.batch_normalize, args.learning_rate, args.gamma, args.tau, args.epsilon, args.hidden_size, args.hidden_n, args.hidden_activation, args.batch_size, args.memory_capacity, args.load_path, args.covariance, args.solve_threshold]

#run experiments
rewards = recursive_experiment(keys, vals, [])
  
#get legend
l_keys = []
l_vals = []
for i,v in enumerate(vals):
  if type(v) == list and len(v) > 1:
    l_keys += [keys[i]]
    l_vals += [v]
legend = recursive_legend(l_keys, l_vals, []) 

#save results
if args.output_path is not None:
  result_file = open(args.output_path, 'w')
  result_file.write(str(args) + '\n')
  result_file.write(str(legend) + '\n')
  result_file.write(str(rewards))
  result_file.close()

#plot
if args.graph: #TODO separate plots for different envs - subplots or completely separate?
  for r in np.mean(rewards, -2): #mean of repeats
    plt.plot(r,'o')
  plt.legend(legend)
  plt.show()
