import sys
import random
import math
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt


from memory import Memory
from model import Model
sys.path.append('../base/')
from minesweeper import Game


class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps, gamma, decay, max_steps):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._gamma = gamma
        self._decay = decay
        self._eps = self._max_eps
        self._max_steps = max_steps
        self._steps = 0
        self._reward_store = []

    def startup(self, n_pre_games):
        for x in range(n_pre_games):
            state = self._env.reset()
            steps = 0
            while True:
                action = random.randint(0, self._model.num_actions - 1)
                next_state, reward, done = self._env.step(action)

                if done:
                    next_state = None

                self._memory.add_sample((state, action, reward, next_state))
                steps += 1
                state = next_state

                if done or steps > self._max_steps:
                    break

    def run(self):
        state = self._env.reset()
        steps = 0
        tot_reward = 0
        while True:
            action = self._choose_action(state)
            next_state, reward, done = self._env.step(action)

            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))

            if self._steps % 5 == 0:
                self._replay()

            # exponentially decay the eps value
            steps += 1
            self._steps += 1
            self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._decay * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done or steps > self._max_steps:
                self._reward_store.append(tot_reward)
                break

        # print("Steps {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(self._sess, state))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([np.zeros(self._model.dimensions) if val[0] is None else val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.dimensions) if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(self._sess, states)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(self._sess, next_states)
        # setup training arrays
        x = np.zeros((len(batch), *self._model.dimensions))
        qs = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
            x[i] = state
            qs[i] = current_q
        self._model.train_batch(self._sess, x, qs)

    @property
    def eps(self):
        return self._eps

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


MAX_EPSILON = 0.25
MIN_EPSILON = 0.001
LAMBDA = 0.000001
GAMMA = 0.99
BATCH_SIZE = 500

STARTUP_GAMES = 2000
NUM_EPISODES = 10000

SIZE_X = 3
SIZE_Y = 3
MINES = 1
MAX_STEPS = 100

TEST_EPISODES = 1000

if __name__ == "__main__":
    env = Game(SIZE_Y, SIZE_X, MINES)

    num_actions = SIZE_Y * SIZE_X

    model = Model(SIZE_Y, SIZE_X, num_actions, BATCH_SIZE)
    saver = tf.train.Saver(max_to_keep=1)
    mem = Memory(50000)

    # Training
    print("Training...")
    with tf.Session() as sess:
        sess.run(model.var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, GAMMA, LAMBDA, MAX_STEPS)
        gr.startup(STARTUP_GAMES)
        print("Startup of {} games finished.".format(STARTUP_GAMES))
        cnt = 1
        while cnt <= NUM_EPISODES:
            if cnt % 100 == 0:
                print('Episode {} of {}'.format(cnt, NUM_EPISODES))
                print('Average reward: {}, eps: {}'.format(np.mean(gr.reward_store[-100:]), gr.eps))
            gr.run()
            cnt += 1

        # Save model for later use and testing
        print("Saving model...")
        saver.save(sess, "../tmp/model")

        # Plot average reward per 100 games over time
        plt.plot(np.convolve(gr.reward_store, np.ones((100,))/100, mode='valid'))
        plt.show()

    # Testing
    print("Testing...")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('../tmp/'))
        test_reward = []
        won_games = 0
        for _ in range(TEST_EPISODES):
            state = env.reset()
            steps = 0
            tot_reward = 0
            while True:
                action = np.argmax(model.predict_one(sess, state))
                next_state, reward, done = env.step(action)

                # is the game complete? If so, set the next state to
                # None for storage sake
                if done:
                    next_state = None

                steps += 1

                # move the agent to the next state and accumulate the reward
                state = next_state
                tot_reward += reward

                # if the game is done, break the loop
                if done or steps > MAX_STEPS:
                    test_reward.append(tot_reward)
                    if env.won() == 1:
                        won_games += 1
                    break

        print("Test Results:")
        print("Average reward: {}".format(np.mean(test_reward)))
        print("Win rate: {}".format(won_games/TEST_EPISODES))

