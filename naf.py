#Based on 'Continuous Deep Q-learning with Model Based Acceleration' by Gu et al, 2016. Available from: https://arxiv.org/pdf/1603.00748.pdf

import gym
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.ops.distributions.util import fill_lower_triangular

def random_process(n):
  mean = np.zeros(n)
  cov = np.eye(n)
  return np.random.multivariate_normal(mean,cov)

def scale_action(a, low, high): #assume domain [-1,1]
  if a > 1:
    a = 1
  elif a < -1:
    a = -1
  return (a+1)*(high-low)/2+low #range [low,high]

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

class Agent:
  def __init__(self, state_n, action_n):
    self.state_n = state_n
    self.action_n = action_n
    self.learning_rate = 0.0001
    self.gamma = 0.90
    self.epsilon = 0.1
    M_n = int((self.action_n)*(self.action_n+1)/2)
    V_n = self.action_n
    h_n = self.action_n
   
    tf.reset_default_graph()
    self.target = tf.placeholder(shape=[1,action_n],dtype=tf.float32,name="target")
    self.x = tf.placeholder(shape=[1,state_n],dtype=tf.float32,name="state")
    self.u = tf.placeholder(shape=[1,action_n],dtype=tf.float32,name="action")
  
    self.w_h = tf.Variable(tf.random_uniform([state_n,h_n],-0.1,0.1))
    self.h = tf.nn.tanh(tf.matmul(self.x,self.w_h)) 
    self.w_M = tf.Variable(tf.random_uniform([h_n,M_n],-0.1,0.1))
    self.M = tf.matmul(self.h,self.w_M)
    self.w_V = tf.Variable(tf.random_uniform([h_n,V_n],-0.1,0.1))
    self.V = tf.matmul(self.h,self.w_V)
    self.w_mu = tf.Variable(tf.random_uniform([h_n,V_n],-0.1,0.1))
    self.mu = tf.nn.tanh(tf.matmul(self.h,self.w_mu))
    self.N = tf.reshape(fill_lower_triangular(self.M), [action_n, action_n])
    self.L = tf.matrix_set_diag(self.N, tf.exp(tf.matrix_diag_part(self.N)))
    self.P = tf.multiply(self.L,self.L) #transpose?
    self.A = (-1.0/2)*tf.matmul(tf.matmul(self.u-self.mu,self.P),tf.matrix_transpose(self.u-self.mu))
    self.Q = tf.add(self.A,self.V)
    self.loss = tf.reduce_sum(tf.square(self.target - self.Q))
    self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
   
    #target network
    self.t_w_h = tf.Variable(tf.random_uniform([state_n,h_n],-0.1,0.1))
    self.t_h = tf.nn.relu(tf.matmul(self.x,self.t_w_h)) 
    self.t_w_V = tf.Variable(tf.random_uniform([h_n,V_n],-0.1,0.1))
    self.t_V = tf.matmul(self.t_h,self.t_w_V)
    self.copy_w_h = self.t_w_h.assign(self.w_h)
    self.copy_w_V = self.t_w_V.assign(self.w_V)

    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)

  def get_action(self,s):
    return self.sess.run(self.mu,feed_dict={self.x:np.reshape(s,[1,-1])})
 
  def get_target(self,a,r,s_next,terminal): 
    return r + self.gamma*self.sess.run(self.t_V,feed_dict={self.x:np.reshape(s_next,[1,-1]),self.u:np.reshape(a,[1,-1])})
   # if terminal:
    #  return r*np.ones(self.action_n)
    #else:  
    #  return r + self.gamma*self.sess.run(self.t_V,feed_dict={self.x:np.reshape(s_next,[1,-1]),self.u:np.reshape(a,[1,-1])})

  def learn(self,s,a,target):
    _,l = self.sess.run([self.optimiser,self.loss],feed_dict={self.x:np.reshape(s,[1,-1]),self.target:np.reshape(target,[1,-1]),self.u:np.reshape(a,[1,-1])})
    return l

  def update_target(self):
    self.sess.run(self.copy_w_h)
    self.sess.run(self.copy_w_V)

interaction_steps = 100
C = 5

capacity = 10000 
batch_size = 32

env = gym.make('InvertedPendulum-v1')

agent = Agent(env.observation_space.shape[0], env.action_space.shape[0])
memory = Memory(capacity, batch_size)

agent.update_target()
plays = 0
while True:
  try:
    rewards = 0
    s = env.reset()
    for i in range(interaction_steps):
      env.render()
      a_unscaled = agent.get_action(s) + agent.epsilon*random_process(env.action_space.shape[0])
      a = scale_action(a_unscaled, env.action_space.low, env.action_space.high)
      if plays % 100 == 0:
        print("a=" + str(a) +  " unscaled=" + str(a_unscaled))
      s_next,r,terminal,_ = env.step(a)
      memory.store([s,a,r,s_next,terminal])
      rewards += r
      s = s_next
      if memory.ready:
        for [t_s,t_a,t_r,t_s_next,t_terminal] in memory.sample():
          target = [agent.get_target(t_a,t_r,t_s_next,t_terminal)]
          agent.learn(t_s, t_a, target)
        if plays % C == 0:
          agent.update_target()
      if terminal:
        epsilon = 1.0/(10 + plays) 
        break
    print("r(" + str(plays) + ")=" + str(rewards))
    plays += 1
  except KeyboardInterrupt:
    break

