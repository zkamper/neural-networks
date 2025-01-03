import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import convolve

def apply_convolution(image):
    kernel = np.ones((3,3))
    convolved = convolve(image, kernel)
    result = np.where(convolved >= 3* 255, 255, image)
    return result

def process_image(image):
    image = image[20:380, :, :]
    image = cv2.resize(image, (72, 72))

    bird_1 = image[:, :, 0]
    bird_1[bird_1 > 227] = 255
    bird_1[bird_1 <= 227] = 0

    bird_2 = image[:, :, 2]
    bird_2[bird_2 > 138] = 255
    bird_2[bird_2 <= 138] = 0

    bird = np.zeros((72, 72))
    bird[(bird_1 == 255) | (bird_2 == 255)] = 255
    bird_mask = apply_convolution(bird)

    pipes = image[:, :, 1]
    pipes[pipes > 136] = 255
    pipes[pipes <=136] = 0

    final_state = pipes
    final_state = np.where(bird_mask == 255, 127, final_state)

    final_state = np.where(final_state == 255, 1, final_state)
    final_state = np.where(final_state == 127, 0.5, final_state)
    # plt.imshow(bird_mask, cmap='gray')
    # plt.show()
    # plt.imshow(pipes, cmap='gray')
    # plt.show()
    # plt.imshow(final_state, cmap='gray')
    # plt.show()

    return final_state