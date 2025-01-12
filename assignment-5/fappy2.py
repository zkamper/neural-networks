import torch
import gymnasium
import flappy_bird_gymnasium
from utils.dqn import DQN
from utils.replay_memory import ReplayMemory
from utils.img_processor import process_image
import numpy as np
import torch.nn.functional as func
import os

def init_weights(l):
    if type(l) == torch.nn.Linear or type(l) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(l.weight)
        l.bias.data.fill_(0.01)

class DQNAgent:
    def __init__(self, input_shape, num_actions, batch_size=32, gamma=0.99, eps=1, eps_min=0.01, eps_decay=0.999, replay_buffer=5_000):
        self.env = gymnasium.make("FlappyBird-v0", render_mode = "rgb_array",use_lidar = False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.policy_net.apply(init_weights)
        self.target_net = DQN(input_shape, num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(replay_buffer)

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # pentru statistici
        self.epoch = np.array([])
        self.rewards = np.array([])
        self.scores = np.array([])

    def choose_action(self, state):
        if np.random.rand() <= self.eps:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_vals = self.target_net(next_states).max(dim=1).values
        expected_q = rewards + self.gamma * next_q_vals * (1. - dones)

        loss = func.mse_loss(q_vals, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self,epochs=5_000,target_update=5):
        max_return = 0
        for epoch in range(epochs):
            self.env.reset()
            state = process_image(self.env.render())
            state = np.stack([state] * 4, axis=0)

            done = False
            ep_return = 0
            score = 0

            while not done:
                action = self.choose_action(state)
                _, reward, done, _, info = self.env.step(action)
                next_state = process_image(self.env.render())
                next_state = np.append(state[1:], [next_state], axis=0)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                ep_return += reward
                score = info['score']
                self.train_batch()

            if ep_return > max_return:
                max_return = ep_return
                torch.save(self.policy_net.state_dict(), f'models/best_model_{ep_return:.1f}.pth')

            self.epoch = np.append(self.epoch, epoch)
            self.rewards = np.append(self.rewards, ep_return)
            self.scores = np.append(self.scores, score)

            if epoch % target_update == 0:
                self.update_target_net()

            if epoch % 20 == 0:
                print(f'Epoch: {epoch}, Return: {ep_return:.1f}')
                np.savez('stats.npz', epoch=self.epoch, rewards=self.rewards, scores=self.scores)

        self.env.close()

if __name__ == '__main__':
    agent = DQNAgent((4, 72, 72), 2, replay_buffer=25_000, gamma=0.95)
    if not os.path.exists('models'):
        os.makedirs('models')
    agent.train(epochs=15_000, target_update=5)