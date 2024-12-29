import matplotlib.pyplot as plt
import cv2

def process_image(image):
    image = image[20:380, :, :]
    image = cv2.resize(image, (72, 72))
    blur = cv2.blur(image, (7, 7))
    pipes = blur[:, :, 0]
    pipes[pipes > 120] = 255
    pipes[pipes < 100] = 0
    bird_mask = image[:, :, 2]
    bird_mask[bird_mask > 142] = 255
    bird_mask[bird_mask <= 142] = 0
    pipes.reshape(72, 72)
    bird_mask.reshape(72, 72)
    # plt.imshow(pipes, cmap='gray')
    # plt.show()
    # plt.imshow(image[:, :, 1], cmap='gray')
    # plt.show()
    # plt.imshow(bird_mask, cmap='gray')
    # plt.show()

    state = image[:, :, 1]
    state[state > 0] = 255
    state.reshape(72, 72)
    return state