import flappy_bird_gymnasium
import gymnasium
import numpy as np

from torch.distributions import Categorical
import torch.optim as optim
import torch
from torch.nn.functional import mse_loss

from utils import img_processor
from utils.Critic import Critic
from utils.actor import Actor

env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
display_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

# env variables
USE_HUMAN_RENDER = True
FRAMES_TO_SKIP_NORMAL = 0
FRAMES_TO_SKIP_JUMP = 10
GOAL_SCORE = 20

# network parameters
EPOCHS = 1000
LEARNING_RATE = 0.001
GAMMA = 0.99


env.reset(seed=42)
display_env.reset(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# advantage actor critic
def train_a2c(actor, critic, gamma=GAMMA, lr_a=LEARNING_RATE, lr_c=LEARNING_RATE, epochs=EPOCHS):
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_a)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_c)

    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    display_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    last_10_scores = torch.zeros(10)
    episode = 0

    first_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0, 0 , 1,0, 0, 0, 0, 0, 0, 0,0, 0 , 0, 0,0,1]

    while torch.min(last_10_scores) < GOAL_SCORE and episode < epochs:
        episode += 1
        seed = np.random.randint(0, 1000)
        env.reset(seed=seed)
        display_env.reset(seed=seed)

        # for action in first_actions:
        #     env.step(action)
        #     display_env.step(action)

        ep_return = 0
        done = False
        state = img_processor.process_image(env.render())
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        epsilon = 0.9
        epsilon_decay_rate = 0.85
        while not done:
            # actor
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            explore_chance = np.random.rand()
            if explore_chance < epsilon:
                action = torch.tensor(env.action_space.sample())
            else:
                action = dist.sample()
            epsilon *= epsilon_decay_rate
            print(epsilon)
            _, reward, done, _, _ = env.step(action.item())

            # skip frames
            if action == 0:
                for _ in range(FRAMES_TO_SKIP_NORMAL):
                    _, reward, done, _, _ = env.step(0)
                    display_env.step(0)
                    if done:
                        break
            else:
                for _ in range(FRAMES_TO_SKIP_JUMP):
                    _, reward, done, _, _ = env.step(0)
                    display_env.step(0)
                    if done:
                        break
            if done:
                break

            next_state = img_processor.process_image(env.render())
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            display_env.step(action.item())

            # critic
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)

            # advantage
            td_target = reward + gamma * next_value * (1. - done)
            advantage = td_target - value

            # critic update cu MSE
            critic_loss = mse_loss(value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # actor update
            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            state_tensor = next_state_tensor
            ep_return += reward

            last_10_scores = torch.cat((last_10_scores[1:], torch.tensor(ep_return).unsqueeze(dim=0)))

        print(f"Episode: {episode}, Score: {ep_return}")
        # save actor model
        if episode % 20 == 0:
            torch.save(actor.state_dict(), f"actor_model_{episode}_{ep_return:.2f}.pt")
    env.close()
    display_env.close()

if __name__ == "__main__":
    actor = Actor(2).to(device)
    critic = Critic(72 * 72).to(device)
    train_a2c(actor, critic)