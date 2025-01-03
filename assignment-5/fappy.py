from argparse import Action
import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import img_processor
from utils import Critic


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
display_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)


FRAMES_TO_SKIP_NORMAL = 0
FRAMES_TO_SKIP_JUMP = 10

env.reset()
display_env.reset()
index = 0

while True:

    action = env.action_space.sample()
    _, reward, terminated, _, info = env.step(action)
    index += 1
    if index == 5:
        image = env.render()
        state = img_processor.process_image(image)
    display_env.step(action)

    if action == 0:
        for _ in range(FRAMES_TO_SKIP_NORMAL):
            _, reward, terminated, _, info = env.step(0)
            display_env.step(0)
            if terminated:
                break
    else:
        for _ in range(FRAMES_TO_SKIP_JUMP):
            _, reward, terminated, _, info = env.step(0)
            display_env.step(0)

            if terminated:
                break
    if terminated:
        break

env.close()