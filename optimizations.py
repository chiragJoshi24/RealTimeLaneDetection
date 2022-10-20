import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# read the image
img1 = cv.imread('img/road_1.jpeg', 0)

# remove noise
img = cv.GaussianBlur(img1, (3, 3), 0)

# convolute with proper kernels
laplacian = cv.Laplacian(img, cv.CV_64F)
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

# Create a custom kernel
# 3x3 array for edge detection

sobel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

# TODO: Create and apply a Sobel x operator
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
filtered_image_y = cv.filter2D(img1, -1, sobel_y)
filtered_image_x = cv.filter2D(img1, -1, sobel_x)
plt.subplot(2, 2, 1), plt.imshow(img1, cmap='gray')
plt.title('Gray scale'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(filtered_image_x, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(filtered_image_y, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()


# canny edge detection
edges = cv.Canny(img, 250, 250)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
