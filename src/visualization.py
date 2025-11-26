import numpy as np
import cv2

class EntropyVisualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width), dtype=np.uint8)

    def generate_bitmap(self, entropy_pool):
        if not entropy_pool:
            return self.image
        bits = np.unpackbits(np.frombuffer(entropy_pool, dtype=np.uint8))
        num_pixels = self.width * self.height
        if len(bits) > num_pixels:
            bits = bits[:num_pixels]
        img_flat = np.zeros(num_pixels, dtype=np.uint8)
        img_flat[:len(bits)] = bits * 255
        self.image = img_flat.reshape((self.height, self.width))
        return self.image

    def generate_histogram(self, entropy_pool):
        hist_height = self.height
        hist_width = self.width
        bin_width = int(np.ceil(hist_width / 256))
        hist_image = np.zeros((hist_height, hist_width), dtype=np.uint8)
        if not entropy_pool:
            return hist_image
        hist = cv2.calcHist([np.frombuffer(entropy_pool, dtype=np.uint8)], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
        for i in range(256):
            x = i * bin_width
            y = int(hist[i])
            cv2.rectangle(hist_image, (x, hist_height - y), (x + bin_width - 1, hist_height), 255, -1)
        return hist_image