import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../rl_agent/')
import qnetwork
sys.path.append('../base/')
from minesweeper import Game

print("Initialize training.")

# Game parameters
field_size = 5
num_actions = field_size*field_size
mines = 3

# Training parameters
batch_size = 100  # How many experiences to use for each training step.
update_freq = 5  # How often to perform a training step.
y = .9  # Discount factor on the target Q-values
num_episodes = 100000  # How many episodes of game environment to train network with.
pre_train_steps = 20000  # How many steps of random actions before training begins.
max_epLength = 50  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "../dqn"  # The path to save our model to.
tau = 0.0001  # Rate to update target network toward primary network

# Start training
tf.reset_default_graph()
mainQN = qnetwork.QNetwork(field_size, num_actions)
targetQN = qnetwork.QNetwork(field_size, num_actions)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = qnetwork.update_target_graph(trainables, tau)

myBuffer = qnetwork.ExperienceBuffer(20000)

# Set the rate of random actions
e = 0.005

# create lists to contain total rewards and steps per episode
jList = []
rList = []
Result_store = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

print("Start training.")
with tf.Session() as sess:
    sess.run(init)
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episodeBuffer = qnetwork.ExperienceBuffer()
        # Reset environment and get first new observation
        game = Game(field_size, field_size, mines)
        s = game.state()
        d = False
        rAll = 0
        j = 0
        actions = []
        # The Q-Network
        while j < max_epLength:  # If the network takes more moves than needed for the field, cancel episode
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, num_actions)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.input: np.reshape(s, [-1, field_size, field_size, 2])})[0]
            y, x = np.unravel_index(a, (field_size, field_size))
            s1, r, d = game.action(y, x, False)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save experience to episode buffer.

            if total_steps > pre_train_steps and total_steps % update_freq:
                trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                # Below we perform the Double-DQN update to the target Q-values
                Q1 = sess.run(mainQN.predict, feed_dict={mainQN.input: np.reshape(np.stack(trainBatch[:, 3]), [-1, field_size, field_size, 2])})
                Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.input: np.reshape(np.stack(trainBatch[:, 3]), [-1, field_size, field_size, 2])})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = trainBatch[:, 2] + (y*doubleQ * end_multiplier)
                # Update the network with our target values.
                _ = sess.run(mainQN.updateModel,
                             feed_dict={mainQN.input: np.reshape(np.stack(trainBatch[:, 0]), [-1, field_size, field_size, 2]),
                                        mainQN.targetQ: targetQ,
                                        mainQN.actions: trainBatch[:, 1]})

                qnetwork.update_target(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1
            actions.append((y, x, False))

            if d:
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        # Periodically display the game.
        if i % 500 == 0:
            print('Display game with moves')
            game.show_field()
            print('Actions:')
            print(actions)
            print('Reward for this game: {}'.format(rAll))
        # Periodically save the model.
#         if i % 1000 == 0:
#             saver.save(sess, path+'/model-'+str(i)+'.ckpt')
#             print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-100:]))
    saver.save(sess, path+'/model-'+str(i)+'.ckpt')
print("Average reward per episodes: " + str(sum(rList)/num_episodes))


# Plotting
rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
plt.show()

#win/lose/unfinished rates
ls_rs = Result_store.append(game.won())
cnt = 1
while cnt <= num_episodes:
    info_int = 200
    if cnt % info_int == 0:
        last_results = ls_rs[-info_int:]
        print('Win/Lose/Unfinished rate: {}, {}, {}'.format(last_results.count(1)/info_int,
                      last_results.count(-1)/info_int, last_results.count(0)/info_int))
cnt += 1
