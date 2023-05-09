import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: rc15/kaggle/sample')
parser.add_argument('--file_path', default='../org-data')
parser.add_argument('--random_seed', type=int, default=62323)
parser.add_argument('--session_size', type=int, default=2)
parser.add_argument('--item_size', type=int, default=5)
parser.add_argument('--max_state_size', type=int, default=10)
parser.add_argument('--fraction', default=[0.8, 0.1, 0.1])

flag_object = parser.parse_args()
np.random.seed(flag_object.random_seed)

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def load_data(flag_object, output_file_path):
    if flag_object.dataset == "kaggle":
        event_df = pd.read_csv(os.path.join(flag_object.file_path, f"kaggle/events.cav"), header=0)
        event_df.columns = ['timestamp', 'session_id', 'behavior', 'item_id', 'transid']
        sampled_session = event_df[event_df['transid'].isnull()]
        sampled_session = sampled_session.drop('transid', axis=1)

        behavior_encoder = LabelEncoder()
        sampled_session['behavior'] = behavior_encoder.fit_transform(sampled_session.behavior)
        sampled_session['is_buy'] = 1 - sampled_session['behavior']
        sampled_session = sampled_session.drop('behavior', axis=1)
        sampled_session = sampled_session.sort_values(by=['session_id', 'timestamp'])


    else:
        if flag_object.dataset == "sample":
            sample_size = 20000
        else:
            sample_size = 3000000

        click_df = pd.read_csv(os.path.join(flag_object.file_path, "rc15/yoochoose-clicks.dat"), header=None, low_memory=False)
        click_df.columns = ['session_id', 'timestamp', 'item_id', 'category']
        click_df['valid_session'] = click_df.session_id.map(click_df.groupby('session_id')['item_id'].size() > 2)
        click_df = click_df.loc[click_df.valid_session].drop('valid_session', axis=1)

        buy_df = pd.read_csv(os.path.join(flag_object.file_path, "rc15/yoochoose-buys.dat"), header=None)
        buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

        sampled_session_id = np.random.choice(click_df.session_id.unique(), sample_size, replace=False)
        sampled_click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]
        sampled_buy_df = buy_df.loc[buy_df.session_id.isin(sampled_click_df.session_id)]

        sampled_click_df = click_df.drop(columns=['category'])
        sampled_buy_df = sampled_buy_df.drop(columns=['price', 'quantity'])

        sampled_click_df['is_buy'] = 0
        sampled_buy_df['is_buy'] = 1

        sampled_session = pd.concat([sampled_click_df, sampled_buy_df], ignore_index=True)
        sampled_session = sampled_session.sort_values(by=['session_id', 'timestamp'])

    to_pickled_df(output_file_path, sampled_session=sampled_session)
    return sampled_session


def filter_data(input_data, session_size, item_size):
    iteration = 0
    while True:
        old_count = input_data.shape[0]
        session_id_df = input_data.groupby("session_id")["timestamp"].count()
        item_id_df = input_data.groupby("item_id")["timestamp"].count()

        session_id_list = session_id_df[session_id_df > session_size].index
        item_id_list = item_id_df[item_id_df > item_size].index

        input_data = input_data.loc[input_data.session_id.isin(session_id_list)]
        input_data = input_data.loc[input_data.item_id.isin(item_id_list)]

        new_count = input_data.shape[0]

        print("iteration", iteration, ":", old_count - new_count)
        iteration += 1
        if new_count == old_count:
            break

    return input_data


def split_data(input_data, frac):
    total_ids = input_data.session_id.unique()
    np.random.shuffle(total_ids)

    frac = np.array(frac)
    train_ids, val_ids, test_ids = np.array_split(total_ids, (frac[:-1].cumsum() * len(total_ids)).astype(int))
    train_sessions = input_data[input_data['session_id'].isin(train_ids)]
    val_sessions = input_data[input_data['session_id'].isin(val_ids)]
    test_sessions = input_data[input_data['session_id'].isin(test_ids)]

    return train_sessions, test_sessions, val_sessions


def pad_history(itemlist, length, pad_item=0):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


def generate_replay_buffer(sessions, length=10):
    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [],[],[]

    groups=sessions.groupby('session_id')
    ids=sessions.session_id.unique()

    for id in ids:
        group=groups.get_group(id)
        history=[]
        for index, row in group.iterrows():
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length)
            a=row['item_id']
            is_b=row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s=list(history)
            len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
            next_s=pad_history(next_s,length)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1]=True

    output_dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,
         'is_done':is_done}
    output_df = pd.DataFrame(data=output_dic)
    return output_df[output_df["len_state"] >=2]


def generate_data_statis(train_replay_buffer,test_replay_buffer, val_replay_buffer, state_size, item_count):
    train_total_purchase = train_replay_buffer[train_replay_buffer["is_buy"] == 1].shape[0]
    train_total_clicks = train_replay_buffer[train_replay_buffer["is_buy"] == 0].shape[0]

    val_total_purchase = val_replay_buffer[val_replay_buffer["is_buy"] == 1].shape[0]
    val_total_clicks = val_replay_buffer[val_replay_buffer["is_buy"] == 0].shape[0]

    test_total_purchase = test_replay_buffer[test_replay_buffer["is_buy"] == 1].shape[0]
    test_total_clicks = test_replay_buffer[test_replay_buffer["is_buy"] == 0].shape[0]

    dic = {'state_size': [state_size], 'item_num': [item_count], "train_total_purchase": [train_total_purchase],
           "train_total_clicks": [train_total_clicks], "val_total_purchase": [val_total_purchase],
           "val_total_clicks": [val_total_clicks], "test_total_purchase": [test_total_purchase],
           "test_total_clicks": [test_total_clicks]}

    return pd.DataFrame(data=dic)


def generate_data_mask(all_session_mask, len_max):
    mask_list = [[1]*len_state+[0]*(len_max-len_state) for len_state in all_session_mask]
    return mask_list


def generate_graph_data(data: np.ndarray):
    state_alias_inputs, state_A, state_items = get_state(data[:, 0])
    state_mask = generate_data_mask(data[:, 1], 10)
    actions = data[:, 2]
    reward = data[:, 3]
    next_state_alias_inputs, next_state_A, next_state_items = get_state(data[:, 4])
    next_state_mask = generate_data_mask(data[:, 5], 10)
    done_ary = np.array([1 if i else 0 for i in data[:, 6]])
    data_dic = {"state_alias_inputs": state_alias_inputs, "state_A": state_A, "state_items":state_items, "state_mask":state_mask, "actions":actions, "reward":reward, "next_state_alias_inputs":next_state_alias_inputs, "next_state_A":next_state_A, "next_state_items":next_state_items, "next_state_mask":next_state_mask, "done_ary":done_ary}
    return pd.DataFrame(data=data_dic)


def get_state(state):
    items, n_node, A, alias_inputs = [], [], [], []

    for u_input in state:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)

    for u_input in state:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

    return alias_inputs, A, items


def main():
    output_file_path = f"../data/{flag_object.dataset}"

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    sampled_session = load_data(flag_object, output_file_path=output_file_path)

    filted_data = filter_data(sampled_session, session_size=flag_object.session_size, item_size=flag_object.item_size)

    item_encoder = LabelEncoder()
    filted_data['item_id'] = item_encoder.fit_transform(filted_data.item_id)
    filted_data["item_id"] += 1

    item_ids = filted_data.item_id.unique()
    item_number = len(item_ids)

    print("session count:", filted_data.groupby("session_id")["timestamp"].count().shape[0])
    print("item count", filted_data.groupby("item_id")["timestamp"].count().shape[0])

    train_sessions, test_sessions, val_sessions = split_data(input_data=filted_data, frac=flag_object.fraction)

    to_pickled_df(output_file_path, sampled_train=train_sessions)
    to_pickled_df(output_file_path, sampled_val=val_sessions)
    to_pickled_df(output_file_path, sampled_test=test_sessions)

    # train_replay_buffer = generate_replay_buffer(train_sessions, length=flag_object.max_state_size)
    # val_replay_buffer = generate_replay_buffer(val_sessions, length=flag_object.max_state_size)
    # test_replay_buffer = generate_replay_buffer(test_sessions, length=flag_object.max_state_size)
    #
    # to_pickled_df(output_file_path, train_replay_buffer=train_replay_buffer)
    # to_pickled_df(output_file_path, val_replay_buffer=val_replay_buffer)
    # to_pickled_df(output_file_path, test_replay_buffer=test_replay_buffer)
    #
    # data_statis = generate_data_statis(train_replay_buffer, val_replay_buffer, test_replay_buffer,
    #                                    state_size=flag_object.max_state_size, item_count=item_number)
    #
    # to_pickled_df(output_file_path, data_statis=data_statis)
    #
    # train_rb = generate_graph_data(train_replay_buffer.values)
    # to_pickled_df(output_file_path, train_rb=train_rb)
    # test_rb = generate_graph_data(test_replay_buffer.values)
    # to_pickled_df(output_file_path, test_rb=test_rb)
    # val_rb = generate_graph_data(val_replay_buffer.values)
    # to_pickled_df(output_file_path, val_rb=val_rb)


if __name__ == '__main__':
    for dataset in {"kaggle", "rc15"}:
        flag_object.dataset = dataset
        main()





