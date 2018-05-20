import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append('../rl_agent/')
import qnetwork
sys.path.append('../base/')
from minesweeper import Game

# Game parameters
field_size = 4
num_actions = field_size*field_size*2
mines = 1

# Training parameters
batch_size = 32  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
annealing_steps = 100000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 10000  # How many steps of random actions before training begins.
max_epLength = 200  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "../dqn"  # The path to save our model to.
tau = 0.001  # Rate to update target network toward primary network

# Start training
tf.reset_default_graph()
mainQN = qnetwork.QNetwork(field_size, num_actions)
targetQN = qnetwork.QNetwork(field_size, num_actions)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = qnetwork.update_target_graph(trainables, tau)

myBuffer = qnetwork.ExperienceBuffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE)/annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

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
                a = sess.run(mainQN.predict, feed_dict={mainQN.input: np.reshape(s, [-1, field_size, field_size, 1])})[0]
            if a >= num_actions//2:
                a -= num_actions//2
                flag = True
            else:
                flag = False
            y, x = np.unravel_index(a, (field_size, field_size))
            s1, r, d = game.action(y, x, flag)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save experience to episode buffer.

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % update_freq == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.input: np.reshape(np.stack(trainBatch[:, 3]), [-1, field_size, field_size, 1])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.input: np.reshape(np.stack(trainBatch[:, 3]), [-1, field_size, field_size, 1])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y*doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.input: np.reshape(np.stack(trainBatch[:, 3]), [-1, field_size, field_size, 1]),
                                            mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})

                    qnetwork.update_target(targetOps,sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1
            actions.append((y, x, flag))

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
        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)
    saver.save(sess, path+'/model-'+str(i)+'.ckpt')
print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")