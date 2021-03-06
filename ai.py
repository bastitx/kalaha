from game import Game
import numpy as np
import tensorflow as tf
import random

num_units = [14, 512, 400, 6]
episodes = 500
gamma = 0.1    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.05
epsilon_decay = 0.994
learning_rate = 0.0001
#alpha = 0.8

x = tf.placeholder(tf.float32, [None, num_units[0]])
actual_y = tf.placeholder(tf.float32, [None, num_units[-1]])

weights = [tf. Variable(tf.random_normal([num_units[i-1], num_units[i]]))
                for i in range(1, len(num_units))]

biases = [tf. Variable(
    tf.zeros([num_units[i]])) for i in range(1, len(num_units))]

layers = [x]
for i in range(len(num_units)-2):
    layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i], weights[i]), biases[i])))
y = tf.matmul(layers[-1], weights[-1]) + biases[-1]
y_probs = tf.nn.softmax(y)

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
        total_rew1 = 0
        total_rew2 = 0
        done = False
        memory = []
        player = 0
        while not done:
            #valid_moves = np.array(env.action_space()) - 1
            #valid_scores = predicted_probs[valid_moves]
            #a = np.random.choice(env.action_space(), 1, p=valid_scores/valid_scores.sum())
            if player == 0:
                predicted_probs = sess.run(y_probs, feed_dict={x: state})[0]
                a = np.argmax(predicted_probs[0]) + 1
                if np.random.rand(1) <= epsilon:
                    a = random.sample(env.action_space(),1)[0]
            else:
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
