from game import Game
import numpy as np
import tensorflow as tf
import random

episodes = 500
gamma = 0.1    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.05
epsilon_decay = 0.994
learning_rate = 0.0001

x = tf.placeholder(tf.float32, [1,14])
actual_y = tf.placeholder(tf.float32, 6)

linear1 = tf.layers.dense(x, 512)
linear1_dropout = tf.layers.dropout(linear1, rate=0.2)
linear1_act = tf.nn.elu(linear1_dropout)
linear2 = tf.layers.dense(linear1_act, 400)
linear2_act = tf.nn.elu(linear2)
lstm = tf.nn.rnn_cell.LSTMCell(400)
lstm_state = tf.placeholder(tf.float32, [1,400])
lstm_scope = tf.placeholder(tf.float32, [1,400])
output_lstm = lstm(linear2_act, lstm_state)
critic_linear = tf.layers.dense(output_lstm, 1)
actor_linear = tf.layers.dense(output_lstm, 6)

cost = tf.reduce_sum(tf.squared_difference(y, actual_y))
#cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=actual_y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    env = Game()
    for i in range(episodes):
        state = env.reset()
        state = np.matrix(state)
        m_state = lstm.zero_state(1)
        m_scope = lstm.zero_state(1)
        total_rew1 = 0
        total_rew2 = 0
        done = False
        memory = []
        player = 0
        while not done:
            logit, value, m_out = sess.run([actor_linear, critic_linear, output_lstm],
                feed_dict= {x: state, lstm_state: m_state, lstm_scope: m_scope})[0]
            valid_moves = np.array(env.action_space()) - 1
            valid_scores = predicted_probs[valid_moves]
            a = np.random.choice(env.action_space(), 1, p=valid_scores/valid_scores.sum())

            a = np.argmax(predicted_probs[0]) + 1
            if np.random.rand(1) <= epsilon:
                a = random.sample(env.action_space(),1)[0]
            prev_state = state
            state, rew, done, next_player = env.step(a)
            if player == 0:
                total_rew1 += rew
            else:
                total_rew2 += rew
            state = np.matrix(state)
            memory += [(player, prev_state, a, rew, state, done)]
            player = next_player

        target1 = 0
        target2 = 0
        #for player, state, a, rew, next_state, done in reversed(memory):
        for player, state, a, rew, next_state, done in memory:
            if player == 1:
                continue
            target_y = sess.run(y, feed_dict={x: state})
            if player == 0:
                target1 = rew + gamma * target1
                target_y[0, a-1] = target1
            else:
                target2 = rew + gamma * target2
                target_y[0, a-1] = target2
            target = rew + gamma * np.amax(sess.run(y, feed_dict={x: next_state}))
            target_y[0, a-1] = target
            sess.run(optimizer, feed_dict={x: state, actual_y: target_y})
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(i, total_rew1, total_rew2, epsilon)


    while True:
        state = env.reset()
        state = np.matrix(state)
        done = False
        player = 0
        while not done:
            if player == 0:
                print rew
                print(state.reshape(2,7))
                a = input("Select a field {0}: ".format(env.action_space()))
            else:
                predicted_y = sess.run(y, feed_dict={x: state})[0]
                a = np.argmax(predicted_y[0]) + 1
                if np.random.rand(1) <= epsilon:
                    a = random.sample(env.action_space(),1)[0]
            prev_state = state
            state, rew, done, next_player = env.step(a)
            player = next_player
            state = np.matrix(state)
