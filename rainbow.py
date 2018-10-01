import argparse
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from atari_wrappers_openai import make_atari, wrap_deepmind
from replay_buffer import *


# Builds a dueling network: two separate full layers, followed by Q and V output layers,
# followed by their amalgamation.
def dueling_layer(input, layer_func, full_size_each, num_actions):
    full_value = layer_func(input, num_outputs=full_size_each, activation_fn=tf.nn.relu)
    full_adv = layer_func(input, num_outputs=full_size_each, activation_fn=tf.nn.relu)
    out_value = layer_func(full_value, num_outputs=1, activation_fn=None)
    out_adv = layer_func(full_adv, num_outputs=num_actions, activation_fn=None)
    return out_value + out_adv - tf.expand_dims(tf.reduce_sum(out_adv, axis=-1) / num_actions, axis=-1)

# Distributional version a dueling network.
def dueling_p_layer(input, layer_func, full_size_each, c):
    full_value = layer_func(input, num_outputs=full_size_each, activation_fn=tf.nn.relu)
    full_adv = layer_func(input, num_outputs=full_size_each, activation_fn=tf.nn.relu)
    out_value = layer_func(full_value, num_outputs=c.num_atoms, activation_fn=None)
    out_value = tf.reshape(out_value, [-1, c.num_atoms, 1])
    out_adv = layer_func(full_adv, num_outputs=c.num_actions * c.num_atoms, activation_fn=None)
    out_adv = tf.reshape(out_adv, [-1, c.num_atoms, c.num_actions])
    # First, aggregate as usual in each atom independently.
    agg = out_value + out_adv - tf.expand_dims(tf.reduce_sum(out_adv, axis=-1) / c.num_actions, axis=-1)
    # Then apply softmax across atoms.
    return tf.nn.softmax(agg, axis=1)

class CartPoleConfig(object):
    def setup(self):
        self.replay_buffer_size = int(2e3)
        self.input_shape = (4,)
        self.env_name = "CartPole-v0"

        self.max_steps = int(1e6)
        self.learning_frequency = 4
        self.batch_size = 32
        if self.use_reward_lookahead:
            self.reward_lookahead = 2
        else:
            self.reward_lookahead = 1
        # NOTE: Shorter update frequencies did not play well with reward_lookahead.
        # Might increase further for more stability?
        self.target_update_freq = 2000
        self.global_gradient_clip = 10.0
        self.learning_starts = 100
        self.gamma = 0.95
        if self.use_noisy_networks:
            self.exploration_max = 0
            self.exploration_min = 0
        else:
            self.exploration_max = 1
            self.exploration_min = 0.01
        self.exploration_anneal_until = 5000
        self.per_alpha = 0.5
        self.per_beta_min = 0.4
        self.per_beta_max = 1.0
        self.adam_learning_rate = 1e-3
        self.adam_epsilon = 1.5e-4
        if self.use_c51:
            self.num_atoms = 51
            self.v_min = -10
            self.v_max = 10
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = np.arange(self.v_min, self.v_max + self.delta_z, self.delta_z)
            assert(len(self.z) == self.num_atoms)

    def encode_obs(self, buf, n):
        return buf.get(n)

    def build_env(self):
        return gym.make(self.env_name)

    def build_q_network(self, input):
        if self.use_noisy_networks:
            layer_func = noisy_layer
        else:
            layer_func = layers.fully_connected

        out = layer_func(input, num_outputs=32, activation_fn=tf.nn.relu)
        if self.use_dueling_networks:
            return dueling_layer(out, layer_func, 32, self.num_actions)
        else:
            out = layer_func(out, num_outputs=32, activation_fn=tf.nn.relu)
            return layer_func(out, num_outputs=self.num_actions, activation_fn=None)

    def build_p_network(self, input):
        if self.use_noisy_networks:
            layer_func = noisy_layer
        else:
            layer_func = layers.fully_connected

        out = layer_func(input, num_outputs=32, activation_fn=tf.nn.relu)
        if self.use_dueling_networks:
            return dueling_p_layer(out, layer_func, 32, self)
        else:
            out = layer_func(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = layer_func(out, num_outputs=self.num_actions * self.num_atoms, activation_fn=None)
            out = tf.reshape(out, [-1, self.num_atoms, self.num_actions])
            return tf.nn.softmax(out, axis=1)

    def build_obs_ph(self):
        obs_t_ph = tf.placeholder(tf.float32, [None] + list(self.input_shape))
        obs_t_float = obs_t_ph
        return obs_t_ph, obs_t_float


class PongConfig(object):
    # We don't do this in __init__ since we first need to set arguments (such as use_prioritized_replay)
    def setup(self):
        self.replay_buffer_size = int(1e6)
        self.input_shape = (84,84,4)
        self.env_name = "PongNoFrameskip-v4"

        self.max_steps = int(1e7)
        self.learning_frequency = 4
        self.batch_size = 32
        if self.use_reward_lookahead:
            self.reward_lookahead = 3
        else:
            self.reward_lookahead = 1
        # Corresponds to period of 32000 frames, since learning_frequency = 4.
        self.target_update_freq = 8000
        self.global_gradient_clip = 10.0
        if self.use_prioritized_replay:
            self.learning_starts = 80000
        else:
            self.learning_starts = 200000
        self.gamma = 0.99
        if self.use_noisy_networks:
            self.exploration_max = 0
            self.exploration_min = 0
        else:
            self.exploration_max = 1
            self.exploration_min = 0.01
        self.exploration_anneal_until = 250000 # NOTE: This overlaps with learning_starts, is this correct?
        self.per_alpha = 0.5
        self.per_beta_min = 0.4
        self.per_beta_max = 1.03
        self.adam_learning_rate = 0.0000625 # From Rainbow paper, \alpha / 4 where \alpha is original DQN LR of 0.00025
        self.adam_epsilon = 1.5e-4
        if self.use_c51:
            self.num_atoms = 51
            self.v_min = -10
            self.v_max = 10
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = np.arange(self.v_min, self.v_max + self.delta_z, self.delta_z)
            assert(len(self.z) == self.num_atoms)

    def encode_obs(self, buf, n):
        return np.stack([buf.get(n + ii) for ii in [-3,-2,-1,0]], axis=-1)

    def build_env(self):
        env = make_atari(self.env_name)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False)
        return SqueezeWrapper(env)        

    def _conv_layer(self, input):
        # NOTE: Input shape matters re: which channels are considered part of the convolution.
        out = layers.convolution2d(input, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu, data_format="NHWC")
        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu, data_format="NHWC")
        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, data_format="NHWC")
        return layers.flatten(out)

    # NOTE: This should not make use of variable reuse, since we'll call multiple times to get distinct networks.
    def build_q_network(self, input):
        if self.use_noisy_networks:
            layer_func = noisy_layer
        else:
            layer_func = layers.fully_connected
        out = self._conv_layer(input)
        if self.use_dueling_networks:
            return dueling_layer(out, layer_func, 256, self.num_actions)
        else:
            out = layer_func(out, num_outputs=512, activation_fn=tf.nn.relu)
            return layer_func(out, num_outputs=self.num_actions, activation_fn=None)

    def build_p_network(self, input):
        if self.use_noisy_networks:
            layer_func = noisy_layer
        else:
            layer_func = layers.fully_connected
        out = self._conv_layer(input)

        if self.use_dueling_networks:
            return dueling_p_layer(out, layer_func, 256, self)
        else:
            out = layer_func(out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layer_func(out, num_outputs=self.num_actions * self.num_atoms, activation_fn=None)
            out = tf.reshape(out, [-1, self.num_atoms, self.num_actions])
            return tf.nn.softmax(out, axis=1)

    def build_obs_ph(self):
        obs_t_ph = tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0
        return obs_t_ph, obs_t_float

# Turn a done sequence (e.g. [0,0,1,0,1]) into a mask on reward values (e.g. [1,1,1,0,0])
# for multi-step q estimates.
def reward_segment_mask_from_done_segment(s):
    mask = []
    current = 1
    for x in s:
        mask.append(current)
        if x > 0:
            current = 0
    return mask

# Takes a done sequence, and returns 0 or 1 depending on whether the multi-step q-estimate
# needs to use a final q-estimate term.
def use_final_q_from_done_segment(s):
    if sum(s) > 0:
        return 0
    else:
        return 1

# From stack overflow.
def get_one_hot(targets, nb_classes):
    targets = np.array(targets)
    res = np.eye(nb_classes)[targets.reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

# The returned observation from the gym wrappers is of shape (84, 84, 1); universally squeeze down to (84, 84)
class SqueezeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, obs):
        return np.squeeze(obs, axis=-1)



def learn(c):
    if c.use_prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(c.encode_obs, c.replay_buffer_size, c.reward_lookahead, c.per_alpha)
    else:
        replay_buffer = SimpleReplayBuffer(c.encode_obs, c.replay_buffer_size, c.reward_lookahead)

    env = c.build_env()
    c.num_actions = env.action_space.n
    last_obs = env.reset()
    
    # Makes debug printing prettier.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    session = tf.Session()

    # NOTE: Forgot to switch this to float32 when testing with CartPole -- tensorflow auto-rounds down to 0.
    # Defers obs placeholders to config, since we need to do something different for different input types.
    obs_t_ph, obs_t_float = c.build_obs_ph()
    act_t_ph = tf.placeholder(tf.int32, [None])
    if c.use_c51:
        m_ph = tf.placeholder(tf.float32, [None, c.num_atoms])
    else:
        q_estimate_ph = tf.placeholder(tf.float32, [None])
    weight_ph = tf.placeholder(tf.float32, [None])

    act_t_onehot = tf.one_hot(act_t_ph, c.num_actions)

    with tf.variable_scope("q_network"):
        if c.use_c51:
            p_network = c.build_p_network(obs_t_float)
        else:
            q_network = c.build_q_network(obs_t_float)
    with tf.variable_scope("q_target_network"):
        if c.use_c51:
            p_target_network = c.build_p_network(obs_t_float)
        else:
            q_target_network = c.build_q_network(obs_t_float)

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network")
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_target_network")

    # Op to copy Q(or p) network to target Q(or p) network.
    update_target_op = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_op.append(var_target.assign(var))
    update_target_op = tf.group(*update_target_op)

    if c.use_c51:
        cross_entropy_error = -tf.reduce_sum(
            m_ph * tf.log(tf.reduce_sum(p_network * tf.expand_dims(act_t_onehot, axis=1), axis=-1)),
            axis=-1)
        per_error = cross_entropy_error
        total_error = tf.reduce_mean(cross_entropy_error)
    else:
        td_error = tf.reduce_sum(q_network * act_t_onehot, axis=-1) - q_estimate_ph
        weighted_squared_error = weight_ph * tf.square(td_error)
        per_error = tf.abs(td_error)
        total_error = tf.reduce_mean(weighted_squared_error)
    optimizer = tf.train.AdamOptimizer(learning_rate=c.adam_learning_rate, epsilon=c.adam_epsilon)
    gradients, variables = zip(*optimizer.compute_gradients(total_error, var_list=q_func_vars))
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, c.global_gradient_clip)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

    t = tf.get_variable("t", dtype=tf.int32, initializer=0)
    t_increment_op = t.assign(t + 1)
    learning_batch_count = 0

    init = tf.global_variables_initializer()
    session.run(init)

    t_val = session.run(t)

    episode_rewards = []
    rewards_this_episode = []
    last_save = t_val

    last_t = t_val
    last_time = time.time()

    while t_val < c.max_steps:
        replay_buffer.add_frame_part1(last_obs)
        if t_val < (c.learning_starts + c.exploration_anneal_until):
            progress = float(t_val - c.learning_starts) / float(c.exploration_anneal_until - c.learning_starts)
            exploration_epsilon = progress * c.exploration_min + (1 - progress) * c.exploration_max
        else:
            exploration_epsilon = c.exploration_min
        if t_val < c.learning_starts or random.random() < exploration_epsilon:
            action = random.randint(0, c.num_actions - 1)
        else:
            if c.use_c51:
                p = session.run(p_network, feed_dict={obs_t_ph: [replay_buffer.encode_recent_obs()]})[0]
                q = np.sum(p * np.expand_dims(c.z, axis=1), axis=0)
            else:
                q = session.run(q_network, feed_dict={obs_t_ph: [replay_buffer.encode_recent_obs()]})[0]
            action = np.argmax(q)

        last_obs, reward, done, info = env.step(action)
        replay_buffer.add_frame_part2(action, reward, done)
        rewards_this_episode.append(reward)

        if done:
            last_obs = env.reset()
            episode_rewards.append(sum(rewards_this_episode))
            rewards_this_episode = []

        if (t_val >= c.learning_starts) and (t_val % c.learning_frequency) == 0 and replay_buffer.can_sample(c.batch_size):
            # Runs on first loop, so target network is always initialized.
            if ((learning_batch_count % c.target_update_freq) == 0):
                print("Updating target")
                session.run(update_target_op)
            learning_batch_count += 1

            obs_t, act_t, obs_tpn, rew_segment, done_segment, indices, weights = replay_buffer.sample(c.batch_size)
            per_beta = (c.per_beta_min * (c.max_steps - t_val) + c.per_beta_max * t_val) / float(c.max_steps)
            weights = weights ** per_beta
            # Weights are normalized by 1 / max weight as in the PER paper.
            weights = weights / weights.max()
            reward_segment_mask = np.array(list(map(reward_segment_mask_from_done_segment, done_segment)))
            use_final_q_mask = np.array(list(map(use_final_q_from_done_segment, done_segment)))
            masked_reward_segment = rew_segment * reward_segment_mask

            if c.use_c51:
                ## Distributional version
                p_network_tpn, p_target_network_tpn = session.run([p_network, p_target_network], {obs_t_ph: obs_tpn})

                if c.use_double_dqn:
                    expected_q = np.full([c.batch_size, c.num_actions], 0.0)
                    for ii in range(c.num_atoms):
                        # p_network shape is [None, num_atoms, num_actions]
                        expected_q += c.z[ii] * p_network_tpn[:,ii,:]
                    a_prime = np.argmax(expected_q, axis=1)
                else:
                    expected_q_target = np.array([c.batch_size, c.num_actions])
                    for ii in range(c.num_atoms):
                        expected_q_target += z[ii] * p_target_network_tpn[:,ii,:]
                    a_prime = np.argmax(expected_q_target, axis=1)
                q_dist_estimate = []
                m = np.full([c.batch_size, c.num_atoms], 0.0)
                for ii in range(c.batch_size):
                    seg = masked_reward_segment[ii]
                    est = np.full([c.num_atoms], 0.0)
                    for jj in range(len(seg)):
                        est += (c.gamma ** jj) * seg[jj]
                    est += (c.gamma ** c.reward_lookahead) * use_final_q_mask[ii] * c.z
                    q_dist_estimate.append(est)
                backup = np.clip(q_dist_estimate, c.v_min, c.v_max)
                b = (backup - c.v_min) / c.delta_z
                lower_b = np.floor(b).astype(np.int32)
                upper_b = np.ceil(b).astype(np.int32)
                for ii in range(c.batch_size):
                    for jj in range(c.num_atoms):
                        m[ii,lower_b[ii,jj]] += p_target_network_tpn[ii,jj,a_prime[ii]] * (upper_b[ii,jj] - b[ii,jj])
                        m[ii,upper_b[ii,jj]] += p_target_network_tpn[ii,jj,a_prime[ii]] * (b[ii,jj] - lower_b[ii,jj])                
                feed_dict = {
                    obs_t_ph: obs_t,
                    act_t_ph: act_t,
                    m_ph: m,
                    weight_ph: weights
                }
            else:
                # Standard version
                q_estimate = []
                q_network_tpn, q_target_network_tpn = session.run([q_network, q_target_network], {obs_t_ph: obs_tpn})
                if c.use_double_dqn:
                    a_prime = np.argmax(q_network_tpn, axis=1)
                    final_q_max = np.sum(q_target_network_tpn * get_one_hot(a_prime, c.num_actions), axis=1)
                else:
                    # NOTE: Huge bug, was using act_tpn here instead of maxing over network.
                    final_q_max = np.amax(q_target_network_tpn, axis=1)
                final_q_contribution = (c.gamma ** c.reward_lookahead) * use_final_q_mask * final_q_max
                # q_estimate = rew_t + (1 - done_t) * gamma * final_q_max
                for ii in range(c.batch_size):
                    seg = masked_reward_segment[ii]
                    est = 0.0
                    for jj in range(len(seg)):
                        est += (c.gamma ** jj) * seg[jj]
                    est += final_q_contribution[ii]
                    q_estimate.append(est)
                feed_dict = {
                    obs_t_ph: obs_t,
                    act_t_ph: act_t,
                    q_estimate_ph: q_estimate,
                    weight_ph: weights
                }

            per_error_update, _ = session.run([per_error, train_op], feed_dict=feed_dict)
            replay_buffer.update_priorities(indices, per_error_update)

        t_val = session.run(t_increment_op)

        report_frequency = 10
        if t_val > c.learning_starts and len(episode_rewards) >= report_frequency:
            print("Summary of last %d episodes" % (report_frequency,))
            print("Max reward: %f" % (max(episode_rewards),))
            print("Min reward: %f" % (min(episode_rewards),))
            print("Avg reward: %f" % (sum(episode_rewards) / len(episode_rewards),))
            print("T: %d" % (t_val,))
            print("T/sec: %d" % ((t_val - last_t) / (time.time() - last_time),))
            last_time = time.time()
            last_t = t_val
            episode_rewards = []


# TODO: Original paper uses factorized noise.
def noisy_layer(input, num_outputs, activation_fn):
    input_size = int(input.shape[1])
    unif_bound = tf.sqrt(1.0 / input_size)
    weight = tf.Variable(
        tf.random_uniform([num_outputs, input_size], minval=-unif_bound, maxval=unif_bound))
    # NOTE: I think initialization is same for bias, although the notation in the paper is for weights only.
    bias = tf.Variable(
        tf.random_uniform([num_outputs], minval=-unif_bound, maxval=unif_bound))
    # NOTE: Could move this to config if desired.
    sigma_naught = 0.5
    sigma_weight = tf.Variable(tf.constant(sigma_naught / input_size, shape=[num_outputs, input_size]))
    sigma_bias = tf.Variable(tf.constant(sigma_naught / input_size, shape=[num_outputs]))
    weight_noise = tf.random_normal([num_outputs, input_size])
    bias_noise = tf.random_normal([num_outputs])
    # NOTE: Apparently matmul doesn't broadcast as expected, so this doesn't work:
    # activation = tf.matmul(weight + sigma_weight * weight_noise, input)
    # This solution due to StackOverflow:
    activation = tf.einsum('oi,ni->no', weight + sigma_weight * weight_noise, input)
    activation += (bias + sigma_bias * bias_noise)
    if activation_fn is not None:
        return activation_fn(activation)
    else:
        return activation

def run(c_class):
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-dist", action="store_true")
    parser.add_argument("--no-per", action="store_true")
    parser.add_argument("--no-double", action="store_true")
    parser.add_argument("--no-lookahead", action="store_true")
    parser.add_argument("--no-duel", action="store_true")
    args = parser.parse_args()
    print(args)

    c = c_class()
    c.use_double_dqn = not(args.no_double)
    c.use_prioritized_replay = not(args.no_per)
    c.use_reward_lookahead = not(args.no_lookahead)
    c.use_noisy_networks = not(args.no_noise)
    c.use_dueling_networks = not(args.no_duel)
    c.use_c51 = not(args.no_dist)

    c.setup()
    learn(c)

if __name__ == "__main__":
    # run(PongConfig)
    run(CartPoleConfig)
