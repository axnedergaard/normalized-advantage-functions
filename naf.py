#Based on 'Continuous Deep Q-learning with Model Based Acceleration' by Gu et al, 2016. Available from: https://arxiv.org/pdf/1603.00748.pdf

#TODO
#parameterise program
#support any type of input? discrete?

import gym
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.ops.distributions.util import fill_lower_triangular
import matplotlib.pyplot as plt
import argparse

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--env',dest='environment',type=str, default='InvertedPendulum-v1')
parser.add_argument('--bn',dest='batch_normalize',type=bool, default=False)
args = parser.parse_args()
p_environment = args.environment
p_batch_normalize = args.batch_normalize

#util functions
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
#  else:   
 #   a = actions
  #  return (a+1)*(high-low)/2+low #range [low,high]

def batch_normalize(x):
  return x

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
    batch_n, in_n = np.shape(x)
    in_n = int(in_n) #TODO not clean...
    if batch_normalize:
      variance_epsilon = 0.000000001
      self.gamma = tf.Variable(tf.constant(1,shape=[in_n],dtype=tf.float32), trainable=True)
      self.beta = tf.Variable(tf.constant(0,shape=[in_n],dtype=tf.float32), trainable=True)
      self.moving_mean = tf.Variable(tf.constant(0,shape=[in_n],dtype=tf.float32), trainable=False)
      self.moving_var = tf.Variable(tf.constant(1,shape=[in_n],dtype=tf.float32), trainable=False)
      mean,var = tf.nn.moments(x, axes=[0])
      update_mean = self.moving_mean.assign(mean) #proper assign... TODO
      update_var = self.moving_mean.assign(var)
      with tf.control_dependencies([update_mean, update_var]):
        x = tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, self.beta, self.gamma, variance_epsilon)

    self.w = tf.Variable(tf.random_uniform([in_n,out_n],-1,1), trainable=True) 
    self.b = tf.Variable(tf.random_uniform([out_n],-1,1), trainable=True) 
    self.z = tf.matmul(x, self.w) + self.b #transpose/order?

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
  def __init__(self, state_n, action_n, batch_n):
    self.state_n = state_n
    self.action_n = action_n
    
    self.learning_rate = 0.0001
    self.gamma = 0.99
    self.tau = 0.1
    self.epsilon = 0.1

    M_n = int((self.action_n)*(self.action_n+1)/2)
    V_n = 1
    mu_n = self.action_n
    h_n = 9
    
    self.batch_size = batch_n

    tf.reset_default_graph()

    #main network
    self.target = tf.placeholder(shape=[None,action_n],dtype=tf.float32,name="target")
    self.x = tf.placeholder(shape=[None,state_n],dtype=tf.float32,name="state")
    self.u = tf.placeholder(shape=[None,action_n],dtype=tf.float32,name="action")

    self.h1 = Layer(self.x, h_n, activation=tf.nn.tanh, batch_normalize=p_batch_normalize)
    self.h2 = Layer(self.h1.h, h_n, activation=tf.nn.tanh, batch_normalize=p_batch_normalize)
    self.M = Layer(self.h2.h, M_n, batch_normalize=p_batch_normalize)
    self.V = Layer(self.h2.h, V_n, batch_normalize=p_batch_normalize)
    self.mu = Layer(self.h2.h, mu_n, activation=tf.nn.tanh, batch_normalize=p_batch_normalize)
    
    self.N = fill_lower_triangular(self.M.h) #done properly for batch?
    self.L = tf.matrix_set_diag(self.N, tf.exp(tf.matrix_diag_part(self.N))) #same as above
    self.P = tf.multiply(self.L,tf.matrix_transpose(self.L))
    self.A = (-1.0/2)*tf.matmul(tf.matmul(tf.reshape(self.u-self.mu.h,[-1,1,self.action_n]),self.P),tf.matrix_transpose(tf.reshape(self.u-self.mu.h,[-1,1,self.action_n])))
    self.Q = self.A + self.V.h
    self.loss = tf.reduce_sum(tf.square(self.target - self.Q))
    self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) 

    #target network
    self.t_h1 = Layer(self.x, h_n, activation=tf.nn.tanh, batch_normalize=p_batch_normalize)
    self.t_h2 = Layer(self.t_h1.h, h_n, activation=tf.nn.tanh, batch_normalize=p_batch_normalize)
    self.t_V = Layer(self.t_h2.h, V_n, batch_normalize=p_batch_normalize)
   
    #main-target update ops
    self.updates = []
    self.updates += [self.t_h1.construct_update(self.h1, self.tau)]
    self.updates += [self.t_h2.construct_update(self.h2, self.tau)]
    self.updates += [self.t_V.construct_update(self.V, self.tau)]

    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)

  def get_action(self,s):
    a = self.sess.run(self.mu.h,feed_dict={self.x:np.reshape(s,[1,-1])})
    return a[0]
 
  def get_target(self,a,r,s_next,terminal): 
    targets = np.reshape(r,[-1,1]) + np.reshape(self.gamma*self.sess.run(self.t_V.h,feed_dict={self.x:s_next,self.u:a}),[-1,1])
    #for i in range(len(terminal)):
    #  if terminal[i]:
    #    targets[i] = r[i]
    return targets

  def learn(self,batch_state,batch_action,batch_target):
    _ = self.sess.run([self.optimiser],feed_dict={self.x:batch_state, self.target:batch_target, self.u:batch_action})

  def update_target(self):
    for update in self.updates:
      self.sess.run(update)

interaction_steps = 10000

capacity = 10000 
batch_size = 32

env = gym.make(p_environment)

agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], batch_size)
memory = Memory(capacity, batch_size)

agent.update_target()
plays = 0
rewards = []
while True:
  try:
    reward = 0
    s = env.reset()
    for i in range(interaction_steps):
      env.render()
      a = agent.get_action(s) + agent.epsilon*random_process(env.action_space.shape[0])
      a = scale_action(a, env.action_space.low, env.action_space.high)
      s_next,r,terminal,_ = env.step(a)
      if i >= interaction_steps-1:
        terminal = True
      memory.store([s,a,r,s_next,terminal])
      reward += r
      s = s_next
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
          batch_action += [t_a[0]] #multi action spaces?
          batch_reward += [t_r]
          batch_terminal += [t_terminal]
        batch_target = agent.get_target(batch_action, batch_reward, batch_state_next, batch_terminal)
        agent.learn(batch_state, batch_action, batch_target)
        agent.update_target()
      if terminal:
        #agent.epsilon = 1.0 / (10 + plays)
        break
    print("r(" + str(plays) + ")=" + str(reward))
    rewards += [reward]
    plays += 1
  except KeyboardInterrupt:
    break

#plt.plot(rewards)
#plt.show()
