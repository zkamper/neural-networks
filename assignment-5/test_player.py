import torch
import gymnasium
import flappy_bird_gymnasium
from utils.dqn import DQN
from utils.img_processor import process_image
import numpy as np

class DQNPlayer:
    def __init__(self, model_path, input_shape, num_actions, seed=42):
        # 2 environments pentru procesare + randare
        self.env_rgb = gymnasium.make("FlappyBird-v0", render_mode="rgb_array",use_lidar = False)
        self.env_human = gymnasium.make("FlappyBird-v0", render_mode="human",use_lidar = False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.env_rgb.reset(seed=seed)
        self.env_human.reset(seed=seed)

        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()

    def play(self, episodes=5):
        for episode in range(episodes):
            seed = np.random.randint(0, 10000)
            state_rgb, _ = self.env_rgb.reset(seed=seed)
            self.env_human.reset(seed=seed)

            frame = self.env_rgb.render()
            if frame is None:
                raise ValueError("Rendered frame is None. Check environment's render_mode.")
            state = process_image(frame)
            state = np.stack([state] * 4, axis=0)  # Stack frames to create input for the DQN

            done = False
            total_reward = 0

            last_score = 0

            while not done:

                self.env_human.render()

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy_net(state_tensor).argmax(dim=1).item()

                _, reward, done, _, _ = self.env_rgb.step(action)
                _, _, _, _, info = self.env_human.step(action)
                # print(info)
                total_reward += reward

                frame = self.env_rgb.render()
                if frame is None:
                    raise ValueError("Rendered frame is None during gameplay.")
                next_state = process_image(frame)
                state = np.append(state[1:], [next_state], axis=0)
                last_score = info['score']

            print(f"Episode {episode + 1}: Total Reward = {total_reward:.1f} Score = {last_score}")

        self.env_rgb.close()
        self.env_human.close()

if __name__ == "__main__":
    model_path = "models/best_model_101.7.pth"
    player = DQNPlayer(model_path, input_shape=(4, 72, 72), num_actions=2, seed=42)
    player.play(episodes=5)
