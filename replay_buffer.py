import numpy as np
import random
from sortedcontainers import SortedList

class CircularBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buf = [None] * size
        self.next_index = 0
        self.number_in_buffer = 0

    def append(self, x):
        index = self.next_index
        self.buf[index] = x
        self.next_index = (index + 1) % self.size
        if self.number_in_buffer < self.size:
            self.number_in_buffer += 1
        return index

    def get(self, k):
        # Allow wrapping.
        k = k % self.size
        if k >= self.number_in_buffer:
            raise IndexError("Index outside circular buffer range.")
        # Allow negatives (after bounds check)
        k = k % self.number_in_buffer
        return self.buf[k]

class SimpleReplayBuffer(object):
    def __init__(self, encode_obs_func, max_size, reward_lookahead):
        self.reward_lookahead = reward_lookahead
        self.encode_obs_func = encode_obs_func
        self.max_size = max_size
        self.obs_buf = CircularBuffer(max_size)
        self.act_buf = CircularBuffer(max_size)
        self.rew_buf = CircularBuffer(max_size)
        self.done_buf = CircularBuffer(max_size)

    # We follow the trick from CS 294 HW3, where you add the observation and responses separately, so that the we can encode an observation stack in between.
    def add_frame_part1(self, obs):
        index = self.obs_buf.append(obs)
        return index

    def add_frame_part2(self, action, reward, done):
        index = self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.done_buf.append(done)
        return index

    def _encode_obs_at_index(self, n):
        # Going to be used for frame-stacking, but made a parameter.
        return self.encode_obs_func(self.obs_buf, n)

    def encode_recent_obs(self):
        return self._encode_obs_at_index((self.obs_buf.next_index - 1) % self.max_size)

    def sample_indices(self, sample_size):
        indices = self.constrain_indices(random.sample(range(self.obs_buf.number_in_buffer), sample_size))
        return indices, ([1.0 / self.obs_buf.number_in_buffer] * sample_size)

    def constrain_indices(self, indices):
        # Indices must have room for reward lookahead and prior frames; if they don't just shift until they do.
        # This distorts probabilities somewhat, but reward lookahead is small compared to buffer size.
        indices2 = []
        for ii in indices:
            if (ii < 3 and self.obs_buf.number_in_buffer < self.max_size):
                # Not enough prior frames to encode.
                indices2.append(3)
            elif ((ii - self.obs_buf.next_index) % self.max_size < 3):
                indices2.append(self.obs_buf.next_index - 1)
            elif ((self.obs_buf.next_index - ii) % self.max_size) <= self.reward_lookahead:
                # Not enough room for reward lookahead.
                indices2.append((self.obs_buf.next_index - self.reward_lookahead - 1) % self.max_size)
            else:
                indices2.append(ii)
        return indices2

    def can_sample(self, sample_size):
        return (self.obs_buf.number_in_buffer - self.reward_lookahead - 4) >= sample_size

    def sample(self, sample_size):
        indices, probs = self.sample_indices(sample_size)

        obs_t = []
        act_t = []
        obs_tpn = []
        rew_segments = []
        done_segments = []
        for ii in indices:
            obs_t.append(self._encode_obs_at_index(ii))
            # print(ii + self.reward_lookahead)
            # print(self.obs_buf.number_in_buffer)
            obs_tpn.append(self._encode_obs_at_index(ii + self.reward_lookahead))
            act_t.append(self.act_buf.get(ii))
            rew_list = []
            done_list = []
            for jj in range(self.reward_lookahead):
                # NOTE: Huge bug, had act_buf here instead of rew_buf
                # Discovered by noticing reward was not always 1 on CartPole-v0
                rew_list.append(self.rew_buf.get(ii + jj))
                done_list.append(self.done_buf.get(ii + jj))
            rew_segments.append(np.array(rew_list))
            done_segments.append(np.array(done_list))

        # Importance sampling weight.
        weight = np.array([1.0 / (self.obs_buf.number_in_buffer * prob) for prob in probs])
        # Indices are unused for SimpleReplayBuffer, but are used to update TD Error in Prioritized Replay.
        return obs_t, act_t, obs_tpn, rew_segments, done_segments, indices, weight

    def update_priorities(self, indices, priorities):
        pass


class SumTreeLeaf(object):
    def __init__(self, parent, value, meta):
        self.parent = parent
        self.value = value
        self.meta = meta

    def insert(self, value, meta):
        new_leaf = SumTreeLeaf(None, value, meta)
        new_node = SumTreeNode(self.parent, self, new_leaf)
        new_leaf.parent = new_node
        self.parent = new_node
        return (new_node, new_leaf)

    def remove(self):
        old_parent = self.parent
        if old_parent is None:
            return (old_parent, None)
        else:
            grandparent = old_parent.parent
            if self == self.parent.left:
                new_parent = self.parent.right
            else:
                new_parent = self.parent.left
            new_parent.parent = grandparent
            if grandparent is not None:
                if grandparent.left == old_parent:
                    grandparent.left = new_parent
                else:
                    grandparent.right = new_parent
                while grandparent is not None:
                    grandparent.value -= self.value
                    grandparent = grandparent.parent
            return (old_parent, new_parent)

    def node_at_value(self, value):
        return self    

    def change_value(self, value):
        remove_upward = self
        old_value = self.value
        while remove_upward is not None:
            remove_upward.value += value - old_value
            remove_upward = remove_upward.parent


    def __repr__(self):
        return "L(%d, %d)" % (self.value, self.meta)

class SumTreeNode(object):
    def __init__(self, parent, left, right):
        self.parent = parent
        self.left = left
        self.right = right
        self.value = left.value + right.value

    def insert(self, value, meta):
        self.value += value
        if self.left is None:
            self.left = SumTreeLeaf(self, value, meta)
            return (self, self.left)
        if self.right is None:
            self.right = SumTreeLeaf(self, value, meta)
            return (self, self.right)
        if self.left.value < self.right.value:
            (new_node, new_leaf) = self.left.insert(value, meta)
            self.left = new_node
            return (self, new_leaf)
        else:
            (new_node, new_leaf) = self.right.insert(value, meta)
            self.right = new_node
            return (self, new_leaf)

    def node_at_value(self, value):
        if value >= self.left.value:
            return self.right.node_at_value(value - self.left.value)
        else:
            return self.left.node_at_value(value)

    def __repr__(self):
        return "N(%d,%s,%s)" % (self.value, repr(self.left), repr(self.right))        


class PrioritizedReplayBuffer(SimpleReplayBuffer):
    def __init__(self, encode_obs_func, max_size, reward_lookahead, alpha = 0.7):
        SimpleReplayBuffer.__init__(self, encode_obs_func, max_size, reward_lookahead)
        self.alpha = alpha
        self.sum_tree = None
        self.node_by_index = [None] * max_size
        # Track the maximal priority seen thus far; insert new elements with this priority.
        self.max_priority_so_far = 1.0
    
    def add_frame_part1(self, obs):
        index = SimpleReplayBuffer.add_frame_part1(self, obs)
        if self.node_by_index[index] is not None:
            (old_parent, new_parent) = self.node_by_index[index].remove()
            if self.sum_tree == old_parent:
                self.sum_tree = new_parent
        if self.sum_tree is None:
            node = SumTreeLeaf(None, self.max_priority_so_far, index)
            self.sum_tree = node
        else:
            new_base, node = self.sum_tree.insert(self.max_priority_so_far, index)
            self.sum_tree = new_base
        self.node_by_index[index] = node

    def add_frame_part2(self, action, reward, done):
        SimpleReplayBuffer.add_frame_part2(self, action, reward, done)
        
    def sample_indices(self, sample_size):
        total = self.sum_tree.value
        segment_size = total / sample_size
        indices = []
        for ii in range(sample_size):
            priority = (ii + random.random()) * segment_size
            node = self.sum_tree.node_at_value(priority)
            indices.append(node.meta)
        indices = self.constrain_indices(indices)
        probs = []
        for ii in indices:
            probs.append(self.node_by_index[ii].value / total)
        return indices, probs

    def update_priorities(self, indices, priorities):
        for ii in range(len(indices)):
            priority = priorities[ii] ** self.alpha
            self.max_priority_so_far = max(priority, self.max_priority_so_far)
            index = indices[ii]
            self.node_by_index[index].change_value(priority)
