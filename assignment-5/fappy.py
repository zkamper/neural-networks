from argparse import Action
import flappy_bird_gymnasium
import gymnasium
import matplotlib.pyplot as plt
import cv2

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

FRAMES_TO_SKIP_NORMAL = 0
FRAMES_TO_SKIP_JUMP = 10

obs, _ = env.reset()
index = 0
while True:

    # retea neuronala 1
    action = env.action_space.sample()

    obs, reward, terminated, _, info = env.step(action)

    if action == 0:
        for _ in range(FRAMES_TO_SKIP_NORMAL):
            obs, reward, terminated, _, info = env.step(0)
            if terminated:
                break
    else:
        for _ in range(FRAMES_TO_SKIP_JUMP):
            obs, reward, terminated, _, info = env.step(0)
            if terminated:
                break
    if terminated:
        break

env.close()