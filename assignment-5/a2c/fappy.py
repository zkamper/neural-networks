import flappy_bird_gymnasium
import gymnasium
import numpy as np

from torch.distributions import Categorical
import torch.optim as optim
import torch
from torch.nn.functional import mse_loss

from utils import img_processor
from utils.critic import Critic, Critic2
from utils.actor import Actor, Actor2

# env variables
USE_HUMAN_RENDER = True
FRAMES_TO_SKIP_NORMAL = 0
FRAMES_TO_SKIP_JUMP = 4
GOAL_SCORE = 40

# network parameters
EPOCHS = 10_000
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON = 0.9
EPSILON_FINAL = 1e-3
EPSILON_DECAY_RATE = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# advantage actor critic
def train_a2c(actor, critic, gamma=GAMMA, lr_a=LEARNING_RATE, lr_c=LEARNING_RATE, epochs=EPOCHS):
    global EPSILON
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_a)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_c)

    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
    display_env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)

    last_10_scores = torch.zeros(10)
    episode = 0

    first_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0, 0 , 1,0, 0, 0, 0, 0, 0, 0,0, 0 , 0, 0,0,1]

    while torch.min(last_10_scores) < GOAL_SCORE and episode < epochs:
        episode += 1
        seed = np.random.randint(0, 1000)
        obs, _ = env.reset(seed=seed)
        display_env.reset(seed=seed)

        # for action in first_actions:
        #     env.step(action)
        #     display_env.step(action)

        ep_return = 0
        done = False
        state = img_processor.process_image(env.render())
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        # state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        while not done:
            # actor
            action_probs = actor(state)
            dist = Categorical(action_probs)
            explore_chance = np.random.rand()
            # if explore_chance < EPSILON:
            #     action = torch.tensor(env.action_space.sample())
            # else:
            action = dist.sample()

            new_obs, reward, done, _, _ = env.step(action.item())
            display_env.step(action.item())
            # skip frames
            if action == 0:
                for _ in range(FRAMES_TO_SKIP_NORMAL):
                    _, reward, done, _, _ = env.step(0)
                    display_env.step(0)
                    ep_return += reward
                    if done:
                        break
            else:
                for _ in range(FRAMES_TO_SKIP_JUMP):
                    _, reward, done, _, _ = env.step(0)
                    display_env.step(0)
                    ep_return += reward
                    if done:
                        break


            next_state = img_processor.process_image(env.render())
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            # next_state = torch.tensor(new_obs, dtype=torch.float32).unsqueeze(0).to(device)

            # train_penalty = max(0, frames_to_penalize - step) * 0.1
            # reward -= train_penalty

            # critic
            value = critic(state)
            next_value = critic(next_state)

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

            state = next_state
            ep_return += reward

            last_10_scores = torch.cat((last_10_scores[1:], torch.tensor(ep_return).unsqueeze(dim=0)))

        # if EPSILON > EPSILON_FINAL:
        #     EPSILON *= EPSILON_DECAY_RATE

        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {ep_return:.1f}")
        # save actor model
        if episode % 50 == 0:
            torch.save(actor.state_dict(), f"actor_model_{episode}_{ep_return:.1f}.pt")
    env.close()
    display_env.close()

if __name__ == "__main__":
    actor = Actor(2).to(device)
    critic = Critic().to(device)
    # actor = Actor2(180, 2).to(device)
    # critic = Critic2(180).to(device)
    train_a2c(actor, critic)