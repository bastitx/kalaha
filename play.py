from game import Game
import numpy as np
import tensorflow as tf

num_units = [14, 512, 400, 400, 6]
x = tf.placeholder(tf.float32, [None, num_units[0]])

weights = [tf.Variable(tf.zeros([num_units[i-1], num_units[i]]))
                for i in range(1, len(num_units))]

biases = [tf.Variable(
    tf.zeros([num_units[i]])) for i in range(1, len(num_units))]

layers = [x]
for i in range(len(num_units)-2):
    layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i], weights[i]), biases[i])))
y = tf.matmul(layers[-1], weights[-1]) + biases[-1]

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './logs')
    env = Game()
    state = env.reset()
    state = np.matrix(state)
    done = False
    player = 0
    while not done:
        if player == 0:
            print(state.reshape(2,7))
            a = input("Select a field {0}: ".format(env.action_space())) - 1
        else:
            predicted_y = sess.run(y, feed_dict={x: state})[0]
            a = np.argmax(predicted_y[0]) + 1
            if np.random.rand(1) <= epsilon:
                a = random.sample(env.action_space(),1)[0]
        prev_state = state
        state, rew, done, next_player = env.step(a)
        player = next_player
        state = np.matrix(state)
