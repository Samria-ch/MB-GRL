import os.path

import torch
from torch import optim

from model import *
from utility import *


class BasedAgent(object):
    def __init__(self, opt, n_node, top_k):
        self.batch_size = opt.batch_size
        self.top_k = top_k
        self.gamma = opt.gamma
        self.target_updata_step = opt.update_step
        self.step = 0

        self.hit_clicks = [0] * len(self.top_k)
        self.ndcg_clicks = [0] * len(self.top_k)
        self.hit_purchase = [0] * len(self.top_k)
        self.ndcg_purchase = [0] * len(self.top_k)

        self.optimizer = None

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.q_loss_fn = nn.MSELoss()

    def top_k_eval(self, sorted_list, actions: torch.Tensor, rewards: torch.Tensor, r_click: float):
        for i in range(len(self.top_k)):
            rec_list = sorted_list[:, -self.top_k[i]:]
            for j in range(len(actions)):
                if actions[j] in rec_list[j]:
                    rank = self.top_k[i] - np.argwhere(rec_list[j] == actions[j])
                    rank = rank.squeeze()
                    if rewards[j] == r_click:
                        self.hit_clicks[i] += 1.0
                        self.ndcg_clicks[i] += 1.0 / np.log2(rank + 1)
                    else:
                        self.hit_purchase[i] += 1.0
                        self.ndcg_purchase[i] += 1.0 / np.log2(rank + 1)

    def compute_acc(self, total_click: int, total_purchase: int):
        hr_click = [number / total_click for number in self.hit_clicks]
        hr_purchase = [number / total_purchase for number in self.hit_purchase]
        ng_click = [number / total_click for number in self.ndcg_clicks]
        ng_purchase = [number / total_purchase for number in self.ndcg_purchase]

        acc_list = []
        acc_list.extend(hr_click)
        acc_list.extend(hr_purchase)
        acc_list.extend(ng_click)
        acc_list.extend(ng_purchase)
        return acc_list

    def start_train(self):
        pass

    def start_val(self):
        pass

    def update_agent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, index, train_data):
        return torch.Tensor(0), None, None, None

    def train_model(self, train_data):
        self.start_train()
        total_loss = 0.0

        batch_index = train_data.generate_batch(self.batch_size, shuffle=True)

        for index in batch_index:
            loss, _,_,_ = self.update(index, train_data)
            self.update_agent(loss)
            total_loss += trans_to_cpu(loss).detach().numpy()

        return total_loss / len(batch_index)

    def eval_model(self, val_data):
        self.start_val()

        total_loss = 0.0

        batch_index = val_data.generate_batch(self.batch_size, shuffle=True)

        for index in batch_index:
            with torch.no_grad():
                loss, _,_,_ = self.update(index, val_data)
            total_loss += trans_to_cpu(loss).detach().numpy()

        return total_loss / len(batch_index)

    def get_acc(self, val_data):
        self.start_val()

        batch_index = val_data.generate_batch(self.batch_size)
        for index in batch_index:
            with torch.no_grad():
                _, score, actions, rewards = self.update(index, val_data)
            score = trans_to_cpu(score).detach().numpy()
            sorted_list = np.argsort(score)

            self.top_k_eval(sorted_list=sorted_list, actions=actions, rewards=rewards, r_click=val_data.r_click)

        acc_array = self.compute_acc(total_click=val_data.total_click, total_purchase=val_data.total_purchase)

        return acc_array


class MbGrlAgent(BasedAgent):
    def __init__(self, opt, n_node, top_k):
        super(MbGrlAgent, self).__init__(opt, n_node, top_k)
        self.model = trans_to_cuda(MbGrl(opt=opt, n_node=n_node))
        self.policy_net = trans_to_cuda(QNetwork(hidden_size=opt.hidden_size, n_node=n_node))
        self.target_net = trans_to_cuda(QNetwork(hidden_size=opt.hidden_size, n_node=n_node))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam([{"params": self.model.parameters()},
                                     {"params": self.policy_net.parameters()}],
                                    lr=opt.lr, weight_decay=opt.l2)

    def double_q_learning(self, state_embedding, actions_tensor, next_state_embedding,
                          reward_tensor, done_tensor):
        q_values = self.policy_net(state_embedding)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        next_q_values = self.policy_net(next_state_embedding)
        next_actions = next_q_values.argmax(1)
        next_q_value = self.target_net(next_state_embedding) \
            .gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q_value = reward_tensor + (1 - done_tensor) * self.gamma * next_q_value

        q_loss = self.q_loss_fn(q_value, expected_q_value)  # Lq

        return q_loss, q_values

    def start_train(self):
        self.model.train()
        self.policy_net.train()
        self.target_net.train()

    def start_val(self):
        self.model.eval()
        self.policy_net.eval()
        self.target_net.eval()

    def update_agent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

        if self.step % self.target_updata_step == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self, i, data: ReplayBuffer):
        state_alias_inputs, state_A, state_items, state_mask, actions, rewards, next_state_alias_inputs, next_state_A, next_state_items, next_state_mask, dones = data.get_slice(
            i)
        actions -= 1

        state_alias_inputs = trans_to_cuda(torch.Tensor(state_alias_inputs).long())
        state_items = trans_to_cuda(torch.Tensor(state_items).long())
        state_A = trans_to_cuda(torch.Tensor(state_A).float())

        next_state_alias_inputs = trans_to_cuda(torch.Tensor(next_state_alias_inputs).long())
        next_state_items = trans_to_cuda(torch.Tensor(next_state_items).long())
        next_state_A = trans_to_cuda(torch.Tensor(next_state_A).float())

        state_mask = trans_to_cuda(torch.Tensor(state_mask).long())
        next_state_mask = trans_to_cuda(torch.Tensor(next_state_mask).long())

        actions_tensor = trans_to_cuda(torch.Tensor(actions).long())
        reward_tensor = trans_to_cuda(torch.Tensor(rewards).long())
        done_tensor = trans_to_cuda(torch.Tensor(dones).long())

        state_item_embedding = self.model.get_item_embedding(state_alias_inputs, state_A, state_items)
        next_state_item_embedding = self.model.get_item_embedding(next_state_alias_inputs, next_state_A, next_state_items)

        session_embedding = self.model.get_session_represent(state_item_embedding, state_mask)
        state_embedding = self.model.get_state_represent(state_item_embedding, state_mask)
        next_state_embedding = self.model.get_state_represent(next_state_item_embedding, next_state_mask)

        q_loss, _ = self.double_q_learning(state_embedding=state_embedding,
                                           actions_tensor=actions_tensor,
                                           next_state_embedding=next_state_embedding,
                                            reward_tensor=reward_tensor, done_tensor=done_tensor)

        score = self.model(session_embedding)
        ce_loss = self.ce_loss_fn(score, actions_tensor)
        loss = q_loss + ce_loss

        return loss, score, actions, rewards

