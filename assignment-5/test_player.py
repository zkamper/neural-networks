import torch
import gymnasium
import flappy_bird_gymnasium
from utils.dqn import DQN
from utils.img_processor import process_image
import numpy as np

class DQNPlayer:
    def __init__(self, model_path, input_shape, num_actions, seed=42):
        # Create two environments: one for processing and one for rendering
        self.env_rgb = gymnasium.make("FlappyBird-v0", render_mode="rgb_array",use_lidar = False)
        self.env_human = gymnasium.make("FlappyBird-v0", render_mode="human",use_lidar = False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the seed for deterministic behavior
        self.env_rgb.reset(seed=seed)
        self.env_human.reset(seed=seed)

        # Initialize the network and load the model
        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()  # Set the network to evaluation mode

    def play(self, episodes=5):
        for episode in range(episodes):
            # Reset both environments with the same seed for synchronization
            seed = np.random.randint(0, 10000)  # Generate a new seed for each episode
            state_rgb, _ = self.env_rgb.reset(seed=seed)
            self.env_human.reset(seed=seed)  # Sync the human-rendered environment

            # Get the initial processed frame
            frame = self.env_rgb.render()  # Render the frame for processing
            if frame is None:
                raise ValueError("Rendered frame is None. Check environment's render_mode.")
            state = process_image(frame)
            state = np.stack([state] * 4, axis=0)  # Stack frames to create input for the DQN

            done = False
            total_reward = 0

            while not done:
                # Render the gameplay to the screen
                self.env_human.render()

                # Predict action using the policy network
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy_net(state_tensor).argmax(dim=1).item()

                # Step both environments (only process results from the RGB environment)
                _, reward, done, _, _ = self.env_rgb.step(action)
                self.env_human.step(action)  # Synchronize the human-rendered environment
                total_reward += reward

                # Process the next frame for internal state updates
                frame = self.env_rgb.render()
                if frame is None:
                    raise ValueError("Rendered frame is None during gameplay.")
                next_state = process_image(frame)
                state = np.append(state[1:], [next_state], axis=0)  # Update the frame stack

            print(f"Episode {episode + 1}: Total Reward = {total_reward:.1f}")

        # Close both environments
        self.env_rgb.close()
        self.env_human.close()

if __name__ == "__main__":
    model_path = "models/best_model_62.6.pth"  # Replace with the path to your saved model
    player = DQNPlayer(model_path, input_shape=(4, 72, 72), num_actions=2, seed=42)
    player.play(episodes=5)
