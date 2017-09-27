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
parser.add_argument('--environment', dest='environment', type=str, default='InvertedPendulum-v1')
parser.add_argument('--repeats', dest='repeats', type=int, default=1)
parser.add_argument('--episodes', dest='episodes', type=int, default=100)
parser.add_argument('--episode_steps', dest='episode_steps', type=int, default=1000)
parser.add_argument('--train_steps', dest='train_steps', type=int, default=5)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, nargs='+', default=0.01)
parser.add_argument('--batch_normalize', dest='batch_normalize', type=bool, default=True)
parser.add_argument('--gamma', dest='gamma', type=float,nargs='+', default=0.99)
parser.add_argument('--tau', dest='tau', type=float,nargs='+', default=0.99)
parser.add_argument('--epsilon', dest='epsilon', type=float, nargs='+', default=0.3)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, nargs='+', default=16)
parser.add_argument('--hidden_n', dest='hidden_n', type=int,nargs='+', default=2)
parser.add_argument('--hidden_activation', dest='hidden_activation', nargs='+', default=tf.nn.relu)
parser.add_argument('--batch_size', dest='batch_size', type=int,nargs='+', default=1024)
parser.add_argument('--memory_capacity', dest='memory_capacity', type=int, nargs='+', default=10000)
parser.add_argument('--verbose', action='store_true') 
args = parser.parse_args()

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
  if args['verbose']:
    print("Experiment " + str(args)) 

  env = gym.make(args['environment'])
 
  experiments_rewards = []
  for i in range(args['repeats']):
    agent = naf.Agent(env.observation_space, env.action_space, args['learning_rate'], args['batch_normalize'], args['gamma'], args['tau'], args['epsilon'], args['hidden_size'], args['hidden_n'], args['hidden_activation'], args['batch_size'], args['memory_capacity'])
    experiment_rewards = []
  
    for j in range(args['episodes']):
      rewards = 0
      state = env.reset()
 
      for k in range(args['episode_steps']):
        if args['render']:
          env.render()
        
        action = agent.get_action(state)
        state_next,reward,terminal,_ = env.step(action)
        
        if k-1 >= args['episode_steps']:
          terminal = True
        agent.observe(state,action,reward,state_next,terminal)

        for l in range(args['train_steps']):
          agent.learn()

        state = state_next
        rewards += reward
        if terminal:
          break
      experiment_rewards += [rewards]
    experiments_rewards += [experiment_rewards]

  return np.mean(experiments_rewards,axis=0)

#main
keys=['verbose', 'graph','render','environment','repeats','episodes','episode_steps','train_steps','batch_normalize', 'learning_rate','gamma','tau','epsilon','hidden_size','hidden_n','hidden_activation','batch_size', 'memory_capacity']
vals=[args.verbose, args.graph, args.render, args.environment, args.repeats, args.episodes, args.episode_steps, args.train_steps, args.batch_normalize, args.learning_rate, args.gamma, args.tau, args.epsilon, args.hidden_size, args.hidden_n, args.hidden_activation, args.batch_size, args.memory_capacity]

rewards=recursive_experiment(keys, vals,[])

#plot
if args.graph:
  #get legend
  l_keys = []
  l_vals = []
  for i,v in enumerate(vals):
    if type(v) == list and len(v) > 1:
      l_keys += [keys[i]]
      l_vals += [v]
  legends = recursive_legend(l_keys, l_vals, []) 
  
  #plot rewards
  for r in rewards:
    plt.plot(r)
  plt.legend(legends)
  plt.show()
