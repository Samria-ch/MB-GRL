import os
import argparse
from matplotlib import pyplot as plt

from tqdm import tqdm
from agent import *
from utility import ReplayBuffer


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: rc15/kaggle/sample')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size')
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--r_click', type=float, default=0.2, help='reward for the click behavior')
parser.add_argument('--r_buy', type=float, default=1, help='reward for the purchase behavior')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--l2', type=float, default=0.00005, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--update_step', type=int, default=2)


def train(epoch, save_file_path, Agent, train_data, val_data):
    train_avg_loss_list = []
    test_avg_loss_list = []

    acc_ary = []
    for _ in tqdm(range(epoch)):
        train_avg_loss_list.append(Agent.train_model(train_data))
        test_avg_loss_list.append(Agent.eval_model(val_data=val_data))

        acc_ary.append(Agent.get_acc(val_data=val_data))
    np.save(os.path.join(save_file_path, "test_acc_array"), np.array(acc_ary, dtype=object))

    plt.figure()
    plt.title("avg_loss")
    plt.plot(range(len(train_avg_loss_list)), train_avg_loss_list, label="train")
    plt.plot(range(len(test_avg_loss_list)), test_avg_loss_list, label="test")
    plt.legend()
    plt.savefig(os.path.join(save_file_path, "loss.jpg"))


def read_data(input_file_path):
    data_statis = pd.read_pickle(os.path.join(input_file_path, 'data_statis.df'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]

    val_total_purchase = data_statis['val_total_purchase'][0]
    val_total_clicks = data_statis['val_total_clicks'][0]

    train_data_df = pd.read_pickle(os.path.join(input_file_path, 'train_rb.df'))
    val_data_df = pd.read_pickle(os.path.join(input_file_path, 'val_rb.df'))

    return train_data_df, val_data_df, \
           {"state_size": state_size, "item_num":item_num, "purchase_count":val_total_purchase, "click_count":val_total_clicks}


if __name__ == '__main__':
    flag_object = parser.parse_args()

    if flag_object.dataset == "kaggle":
        data_directory = "../data/kaggle"
    elif flag_object.dataset == "rc15":
        data_directory = "../data/rc15"
    else:
        data_directory = "../data/sample"

    train_data_df, val_data_df, data_statis = read_data(data_directory)
    val_data = ReplayBuffer(val_data_df, state_size=data_statis["state_size"],
                             total_click=data_statis["click_count"],total_purchase=data_statis["purchase_count"],
                             r_click=flag_object.r_click, r_buy=flag_object.r_buy)

    train_data = ReplayBuffer(train_data_df, state_size=data_statis["state_size"],
                              r_click=flag_object.r_click, r_buy=flag_object.r_buy,
                              total_click=0, total_purchase=0)

    top_k = [5, 10, 20]
    save_directory = "../log/transform_{}".format(flag_object.dataset)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    Agent = MbGrlAgent(opt=flag_object, n_node=data_statis['item_num'], top_k=top_k)

    train(epoch=flag_object.epoch, Agent=Agent, save_file_path=save_directory, train_data=train_data, val_data=val_data)
