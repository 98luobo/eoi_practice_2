import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import line_profiler

torch.set_printoptions(profile="full")


class EOI_Net(nn.Module):
    def __init__(self, obs_len, n_agent):
        super(EOI_Net, self).__init__()
        self.fc1 = nn.Linear(obs_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_agent)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y), dim=1)
        return y


class IVF(nn.Module):
    def __init__(self, obs_len, n_action):
        super(IVF, self).__init__()
        self.fc1 = nn.Linear(obs_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_action)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class EOI_Trainer(object):
    def __init__(self, eoi_net, ivf, ivf_tar, n_agent, n_feature):
        super(EOI_Trainer, self).__init__()
        self.gamma = 0.92
        self.tau = 0.995
        self.n_agent = n_agent
        self.n_feature = n_feature
        # print("N", self.n_feature)  # 30
        self.eoi_net = eoi_net
        self.ivf = ivf
        self.ivf_tar = ivf_tar
        self.optimizer_eoi = optim.Adam(self.eoi_net.parameters(), lr=0.0001)
        self.optimizer_ivf = optim.Adam(self.ivf.parameters(), lr=0.0001)

    def train(self, O, O_Next, A, D):
        O = torch.Tensor(O).cuda()
        O_Next = torch.Tensor(O_Next).cuda()
        A = torch.Tensor(A).cuda().long()
        D = torch.Tensor(D).cuda()

        X = O_Next[:, 0:self.n_feature]
        Y = O_Next[:, self.n_feature:self.n_feature + self.n_agent]  # id
        p = self.eoi_net(X)  # agent i  p(·|ot)
        # CrossEntropy Loss
        # H(p, q) = -sum(p(xi)log(q(xi)))
        loss_1 = -(Y * (torch.log(p + 1e-8))).mean() - 0.1 * (p * (torch.log(p + 1e-8))).mean()
        # print("loss", loss_1)
        self.optimizer_eoi.zero_grad()
        loss_1.backward()
        self.optimizer_eoi.step()

        I = O[:, self.n_feature:self.n_feature + self.n_agent].argmax(axis=1, keepdim=True).long()

        # O[:, 30:30+3]:One_hot(id)
        # print("I", I)#
        r = self.eoi_net(O[:, 0:self.n_feature]).gather(dim=-1, index=I)
        # print("r", r)
        # print("r_size", r.size())  # [256, 1]

        q_intrinsic = self.ivf(O)
        #
        # print("q_ins", q_intrinsic)
        # print("q_inss", q_intrinsic.size())  # [256, 9]
        tar_q_intrinsic = q_intrinsic.clone().detach()
        next_q_intrinsic = self.ivf_tar(O_Next).max(axis=1, keepdim=True)[0]
        # print("next", next_q_intrinsic)
        # print("n_N", next_q_intrinsic.size())  # [256, 1]
        next_q_intrinsic = r * 10 + self.gamma * (1 - D) * next_q_intrinsic
        # print("next", next_q_intrinsic)
        # print(next_q_intrinsic.size())  # [256,1]
        tar_q_intrinsic.scatter_(dim=-1, index=A, src=next_q_intrinsic)
        # A.scatter_(dim, index, B) # 基本用法, tensor A 被就地scatter到 tensor B
        # print("tar", tar_q_intrinsic)
        # print(tar_q_intrinsic.size())  # [256,9]
        loss_2 = (q_intrinsic - tar_q_intrinsic).pow(2).mean()
        # print("LOSS", loss_2.size())
        self.optimizer_ivf.zero_grad()
        loss_2.backward()
        self.optimizer_ivf.step()

        with torch.no_grad():
            for p, p_targ in zip(self.ivf.parameters(), self.ivf_tar.parameters()):
                p_targ.data.mul_(self.tau)
                p_targ.data.add_((self.tau) * p.data)


class EOI_Trainer_Wrapper(object):
    def __init__(self, eoi_trainer, n_agent, n_feature, max_step, batch_size):
        super(EOI_Trainer_Wrapper, self).__init__()
        self.batch_size = batch_size
        self.n_agent = n_agent
        self.o_t = np.zeros((batch_size * n_agent * (max_step + 1), n_feature + n_agent))
        # print("self.ot", len(self.o_t))  # self.ot 5856
        self.next_o_t = np.zeros((batch_size * n_agent * (max_step + 1), n_feature + n_agent))
        self.a_t = np.zeros((batch_size * n_agent * (max_step + 1), 1), dtype=np.int32)
        self.d_t = np.zeros((batch_size * n_agent * (max_step + 1), 1))
        self.eoi_trainer = eoi_trainer

    def train_batch(self, episode_sample):
        episode_obs = np.array(episode_sample["obs"])
        # print("bs", self.batch_size)  # 32
        # print("episode_obs", len(episode_obs))  # 32
        # print(episode_obs.shape[1])  # 61
        episode_actions = np.array(episode_sample["actions"])
        # print("episode_actions", len(episode_actions))  # 32
        episode_terminated = np.array(episode_sample["terminated"])
        ind = 0
        for k in range(self.batch_size):
            for j in range(episode_obs.shape[1] - 2):  # 59
                for i in range(self.n_agent):
                    agent_id = np.zeros(self.n_agent)
                    agent_id[i] = 1
                    self.o_t[ind] = np.hstack((episode_obs[k][j][i], agent_id))
                    self.next_o_t[ind] = np.hstack((episode_obs[k][j + 1][i], agent_id))
                    self.a_t[ind] = episode_actions[k][j][i]
                    self.d_t[ind] = episode_terminated[k][j]
                    ind += 1  # i*j*k = 3*59*32=5664
                if self.d_t[ind - 1] == 1:
                    break
        for k in range(int((ind - 1) / 256)):  # 0-2560
            # print(k)  #
            # 0
            # 1
            # 2
            # 3
            # 4
            # 5
            # 6
            # 7
            # 8
            # 9
            self.eoi_trainer.train(self.o_t[k * 256:(k + 1) * 256], self.next_o_t[k * 256:(k + 1) * 256], self.a_t[k * 256:(k + 1) * 256], self.d_t[k * 256:(k + 1) * 256])

            # sys.exit("Oh")
        # bs 32
        # episode_obs 32
        # 61
        # episode_actions 32
        # Oh
