# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np 
import random
from collections import deque
# As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

# Hyper Parameters for DDPG
LR_A = 0.00001#0.001    # learning rate for actor
LR_C = 0.00002#0.002    # learning rate for critic
GAMMA = 0.8     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64#32
#TARGET_UPDATE_RATE = 0.01 
'''
class DQNPolicy:
    def __init__(self, env,model_path):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("car_saved_networks")

        global summary_writer
        summary_writer = tf.summary.FileWriter('~/car_logs',graph=self.session.graph)

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2


    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        tf.summary.scalar("loss",self.cost)
        global merged_summary_op
        merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        self.time_step += 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > 2000:
            self.train_Q_network()

    def train_Q_network(self):
        
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                    y_batch.append(reward_batch[i])
            else :
                    y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })
        summary_str = self.session.run(merged_summary_op,feed_dict={
        self.y_input : y_batch,
        self.action_input : action_batch,
        self.state_input : state_batch
        })
        summary_writer.add_summary(summary_str,self.time_step)

        # save network every 1000 iteration
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'car_saved_networks/' + 'network' + '-dqn', global_step = self.time_step)

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        if self.time_step > 2000:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def predict(self,state):
        #print('dqn predict shape')
        #print(state.shape)
        #print({self.state_input:state})
        act_pre=np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            }))

        #print('predict_action')
        #print(act_pre)
        return act_pre

    def predict_q(self, state):
        #print(state)
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:state
            })
        #print('predict_q')
        #print(Q_value)
        return Q_value
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)


'''


class DQNPolicy(object):#actually DDPG
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        self.q = q
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
    def predict(self, s):
        a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        return a
    def predict_q(self, s):
        q = self.sess.run(self.q, {self.S: s[np.newaxis, :]})
        #print('predict_q')
        #print(q)
        return q
    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(net, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 256, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 256, activation=tf.nn.relu, name='l5', trainable=trainable)
            a = tf.layers.dense(l5, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 70
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            
            
            
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

