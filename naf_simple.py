#Based on 'Continuous Deep Q-learning with Model Based Acceleration' by Gu et al, 2016. Available from: https://arxiv.org/pdf/1603.00748.pdf
#Simplified advantage function

#TODO
#support any type of input? discrete? unpack->repack
#confirm batch triangular filling

import gym
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.ops.distributions.util import fill_lower_triangular
import matplotlib.pyplot as plt
import argparse

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment',dest='environment',type=str, default='InvertedPendulum-v1')
parser.add_argument('--batch_normalize',action='store_true')
parser.add_argument('--learning_rate',dest='learning_rate', type=float, default=0.01)
parser.add_argument('--batch_size',dest='batch_size',type=int, default=1024)
parser.add_argument('--capacity',dest='capacity',type=int, default=10000)
parser.add_argument('--tau',dest='tau',type=float, default=0.99)
parser.add_argument('--gamma',dest='gamma',type=float, default=0.99)
parser.add_argument('--epsilon',dest='epsilon',type=float, default=0.1)
parser.add_argument('--hidden_size',dest='hidden_size',type=int, default=16)
parser.add_argument('--hidden_n',dest='hidden_n',type=int, default=2)
parser.add_argument('--do_graph',action='store_true')
parser.add_argument('--do_render',action='store_true')
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--hidden_activation',dest='hidden_activation',default=tf.nn.relu)
parser.add_argument('--train_steps',dest='train_steps',type=int, default=5)
parser.add_argument('--post_play',action='store_true')
parser.add_argument('--play_steps',dest='play_steps',type=int,default=1000)
parser.add_argument('--train_Steps',dest='train_steps',type=int,default=5)
args = parser.parse_args()

def random_process(n):
  mean = np.zeros(n)
  cov = np.eye(n)
  return np.random.multivariate_normal(mean,cov)

def scale_action(actions, low, high): #assume domain [-1,1]
  actions = np.clip(actions, -1, 1)
  scaled_actions = []
  for a in actions:
    scaled_actions += [(a+1)*(high-low)/2+low]
  return scaled_actions

def construct_soft_copy(from_vars,to_vars,tau):
  copy = []
  for x,y in zip(from_vars,to_vars):
    copy += [y.assign(tau*x + (1-tau)*y)]
  return copy

class Memory:
  def __init__(self, capacity, batch_size):
    self.m = []
    self.ready = 0
    self.full = 0
    self.capacity = capacity
    self.batch_size = batch_size
 
  def store(self,d):    
    [s,a,r,s_next,terminal] = d
    self.m.append([s,a,r,s_next,terminal])
    if self.full:
      self.m.pop(0)
    if not self.ready and len(self.m) >= self.batch_size:
      self.ready = 1
      print("[Memory ready]")

  def sample(self):
    return random.sample(self.m, self.batch_size)

class Layer:
  def __init__(self, input_layer, out_n, activation=None, batch_normalize=False):
    x = input_layer
    batch_size, in_n = np.shape(x)
    in_n = int(in_n) #TODO cleaner...
    if batch_normalize:
      variance_epsilon = 0.000001
      decay = 0.999
      self.gamma = tf.Variable(tf.constant(1,shape=[in_n],dtype=tf.float32), trainable=True)
      self.beta = tf.Variable(tf.constant(0,shape=[in_n],dtype=tf.float32), trainable=True)
      self.moving_mean = tf.Variable(tf.constant(0,shape=[in_n],dtype=tf.float32), trainable=False)
      self.moving_var = tf.Variable(tf.constant(1,shape=[in_n],dtype=tf.float32), trainable=False)
      mean,var = tf.nn.moments(x, axes=[0])
      update_mean = self.moving_mean.assign(decay*self.moving_mean + (1-decay)*mean) 
      update_var = self.moving_mean.assign(decay*self.moving_var + (1-decay)*var)
      with tf.control_dependencies([update_mean, update_var]):
        x = tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, self.beta, self.gamma, variance_epsilon)

    self.w = tf.Variable(tf.random_uniform([in_n,out_n],-0.1,0.1), trainable=True) 
    self.b = tf.Variable(tf.random_uniform([out_n],-0.1,0.1), trainable=True) 
    self.z = tf.matmul(x, self.w) + self.b #transpose-order?

    if activation is not None:
      self.h = activation(self.z)
    else:
      self.h = self.z

    self.variables = [self.w, self.b]
    if batch_normalize:
      self.variables += [self.gamma, self.beta, self.moving_mean, self.moving_var]
   

  def construct_update(self, from_layer, tau):
    update = []
    for x,y in zip(self.variables, from_layer.variables):
      update += [x.assign(x*tau + (1-tau)*y)]
    return update

class Agent:
  def __init__(self, state_n, action_n):
    self.state_n = state_n
    self.action_n = action_n
    
    self.learning_rate = args.learning_rate
    self.gamma = args.gamma
    self.tau = args.tau
    self.epsilon = args.epsilon
    batch_normalize = args.batch_normalize

    hidden_activation = args.hidden_activation
    H_layer_n = args.hidden_n
    H_n = args.hidden_size 
    M_n = int((self.action_n)*(self.action_n+1)/2)
    V_n = 1
    mu_n = self.action_n

    tf.reset_default_graph()

    #neural network architecture
    self.x = tf.placeholder(shape=[None,state_n],dtype=tf.float32,name="state")
    self.u = tf.placeholder(shape=[None,action_n],dtype=tf.float32,name="action")
    self.target = tf.placeholder(shape=[None,action_n],dtype=tf.float32,name="target")

    self.H = Layer(self.x, H_n, activation=hidden_activation, batch_normalize=batch_normalize) 
    self.t_H = Layer(self.x, H_n, activation=hidden_activation, batch_normalize=batch_normalize) #target
    self.updates = self.t_H.construct_update(self.H, self.tau)
    for i in range(1,H_layer_n):
      self.H = Layer(self.H.h, H_n, activation=hidden_activation, batch_normalize=batch_normalize) 
      self.t_H = Layer(self.t_H.h, H_n, activation=hidden_activation, batch_normalize=batch_normalize) 
      self.updates += self.t_H.construct_update(self.H, self.tau)

    self.M = Layer(self.H.h, M_n, batch_normalize=batch_normalize)
    self.V = Layer(self.H.h, V_n, batch_normalize=batch_normalize)
    self.t_V = Layer(self.t_H.h, V_n, batch_normalize=batch_normalize)  #target
    self.updates += self.t_V.construct_update(self.V, self.tau)
    self.mu = Layer(self.H.h, mu_n, activation=tf.nn.tanh, batch_normalize=batch_normalize)
    
    self.A = -tf.square(self.u - self.mu.h)
    self.Q = self.A + self.V.h
    self.loss = tf.reduce_sum(tf.square(self.target - self.Q))
    self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) 

    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)

  def get_action(self,s):
    a = self.sess.run(self.mu.h,feed_dict={self.x:np.reshape(s,[1,-1])})
    return a[0]
 
  def get_target(self,a,r,s_next,terminal): 
    targets = np.reshape(r,[-1,1]) + np.reshape(self.gamma*self.sess.run(self.t_V.h,feed_dict={self.x:s_next,self.u:a}),[-1,1])
    for i in range(len(terminal)):
     if terminal[i]:
       targets[i] = r[i]
    return targets

  def learn(self,batch_state,batch_action,batch_target):
    a,l,_ = self.sess.run([self.A, self.loss,self.optimiser],feed_dict={self.x:batch_state, self.target:batch_target, self.u:batch_action})
    return a,l

  def update_target(self):
    for update in self.updates:
      self.sess.run(update)

play_steps = args.play_steps
train_steps = args.train_steps

capacity = args.capacity
batch_size = args.batch_size

env = gym.make(args.environment)

agent = Agent(env.observation_space.shape[0], env.action_space.shape[0])
memory = Memory(capacity, batch_size)

agent.update_target()
plays = 0
rewards = []
while True:
  try:
    reward = 0
    s = env.reset()
    for i in range(play_steps):
      if args.do_render:
        env.render()
      a_raw = agent.get_action(s) 
      a_noise = agent.epsilon*random_process(env.action_space.shape[0])
      a = a_raw + a_noise
      a_scaled = scale_action(a, env.action_space.low, env.action_space.high)
      if args.verbose:
        print("Action: " + str(a_raw) + " + " +  str(a_noise) + " = " + str(a) + " -> " + str(a_scaled))
      s_next,r,terminal,_ = env.step(a_scaled)
      if i >= play_steps-1:
        terminal = True
      memory.store([s,a,r,s_next,terminal])
      reward += r
      s = s_next
      for j in range(train_steps):
        if memory.ready: 
          batch_target = []
          batch_state = []
          batch_action = []
          batch_reward = []
          batch_state_next = []
          batch_terminal = []
          for [t_s,t_a,t_r,t_s_next,t_terminal] in memory.sample():
            batch_state_next += [t_s_next]
            batch_state += [t_s]
            batch_action += [t_a] 
            batch_reward += [t_r]
            batch_terminal += [t_terminal]
          batch_target = agent.get_target(batch_action, batch_reward, batch_state_next, batch_terminal)
          advantage,l = agent.learn(batch_state, batch_action, batch_target)
          agent.update_target()
          if args.verbose:
            print("Target: " + str(batch_target))
            print("Loss: " + str(l))
            print("Advantage: " + str(advantage))
      if terminal:
        #agent.epsilon = 1.0 / (1 + plays)
        break
    print("r(" + str(plays) + ")=" + str(reward))
    rewards += [reward]
    plays += 1
  except KeyboardInterrupt:
    break

if args.post_play:
  while True:
    try:
      reward = 0
      s = env.reset()
      for i in range(play_steps):
        env.render()
        a = scale_action(agent.get_action(s), env.action_space.low, env.action_space.high)
        s_next,r,terminal,_ = env.step(a)
        reward += r
        s = s_next
        if terminal:
          break
      print("pr(" + str(i) + ")=" + str(reward))
    except KeyboardInterrupt:
      break
    

if args.do_graph:
  plt.plot(rewards)
  plt.show()
