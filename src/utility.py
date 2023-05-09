import math
import pandas as pd
import torch
import numpy as np


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class ReplayBuffer():
    def __init__(self, input_data: pd.DataFrame, state_size: int,r_click:float, r_buy:float,total_click:int,total_purchase):
        self.r_click = r_click
        self.r_buy = r_buy
        self.total_click = total_click
        self.total_purchase = total_purchase

        self.state_alias_input_ary = np.array(input_data["state_alias_inputs"].to_list(), dtype=int)
        self.state_A_ary = np.array(input_data["state_A"].to_list(), dtype=float)
        self.state_item_ary = np.array(input_data["state_items"].to_list(), dtype=int)
        self.state_mask_ary = np.array(input_data["state_mask"].to_list(), dtype=int)

        self.next_state_alias_input_ary = np.array(input_data["next_state_alias_inputs"].to_list(), dtype=int)
        self.next_state_A_ary = np.array(input_data["next_state_A"].to_list(), dtype=float)
        self.next_state_item_ary = np.array(input_data["next_state_items"].to_list(), dtype=int)
        self.next_state_mask_ary = np.array(input_data["next_state_mask"].to_list(), dtype=int)

        self.action_ary = input_data["actions"].values.astype(dtype=int)
        self.reward_ary = np.array([r_buy if i else r_click for i in input_data["reward"].values])
        self.done_ary = input_data["done_ary"].values

        self.len_max = state_size
        self.length = input_data.shape[0]

    def generate_batch(self, batch_size, shuffle=False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)

            self.state_alias_input_ary = self.state_alias_input_ary[shuffled_arg]
            self.state_A_ary = self.state_A_ary[shuffled_arg]
            self.state_item_ary = self.state_item_ary[shuffled_arg]
            self.state_mask_ary = self.state_mask_ary[shuffled_arg]
            self.action_ary = self.action_ary[shuffled_arg]
            self.reward_ary = self.reward_ary[shuffled_arg]
            self.next_state_alias_input_ary = self.next_state_alias_input_ary[shuffled_arg]
            self.next_state_A_ary = self.next_state_A_ary[shuffled_arg]
            self.next_state_item_ary = self.next_state_item_ary[shuffled_arg]
            self.next_state_mask_ary = self.next_state_mask_ary[shuffled_arg]
            self.done_ary = self.done_ary[shuffled_arg]

        n_batch = math.ceil(self.length / batch_size)
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        state_alias_inputs = self.state_alias_input_ary[i]
        state_A = np.stack(self.state_A_ary[i])
        state_items = self.state_item_ary[i]
        next_state_alias_inputs = self.next_state_alias_input_ary[i]
        next_state_A = np.stack(self.next_state_A_ary[i])
        next_state_items = self.next_state_item_ary[i]
        state_mask = self.state_mask_ary[i]
        next_state_mask = self.next_state_mask_ary[i]

        reward = self.reward_ary[i]
        action = self.action_ary[i]
        done = self.done_ary[i]

        return state_alias_inputs, state_A, state_items, state_mask, action, reward,next_state_alias_inputs, next_state_A, next_state_items, next_state_mask, done


