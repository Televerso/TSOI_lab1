import random
from random import sample

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from scipy.fft import fft2, ifft2
import os

image_orig = cv2.imread('img1.png')

def remove_outliers(image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

show_image(image_orig)
print(image_orig[100,100,:])

def print_pixel_value(image):
    print(image[100,100])

def to_grayscale(image):
    image_gray = np.zeros([image.shape[0],image.shape[1],image.shape[2]], dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_gray[i][j][0:3] = np.sum(image[i][j])//3
    return image_gray

image_gray = to_grayscale(image_orig)
show_image(image_gray)

# p=0.14
def additive_noise_func(image, interval):
    p = 0.14
    image_noise = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.random() < p:
                noise = int(random.uniform(interval[0], interval[1]))
                image_noise[i][j] = image[i,j] + [noise for i in range(image.shape[2])]
            else:
                image_noise[i][j] = image[i,j]

    image_noise = remove_outliers(image_noise)
    return image_noise

image_noise = additive_noise_func(image_gray, [-140,140])
show_image(image_noise)


def divide_by_kernel(image, kernel):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = image[i, j] // (kernel.shape[0] * kernel.shape[1])
    return image


def convolution_by_map(image, kernel):
    matr_a = np.array(image, dtype=float)
    matr_b = np.array(kernel, dtype=float)
    image_conv = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=int)

    for i in range(image_conv.shape[0]):
        for j in range(image_conv.shape[1]):
            s = 0
            for n in range(kernel.shape[0]):
                for m in range(kernel.shape[1]):
                    s += matr_a[i-n,j-m]*matr_b[n,m]
            image_conv[i][j] = s
    return image_conv


def fft_shift(image):
    for i in range(image.shape[0]):
        image[i,:] = np.fft.fftshift(image[i,:])
    for i in range(image.shape[1]):
        image[:,i] = np.fft.fftshift(image[:,i])

def convolution_fft(image, kernel):
    image_fft = np.zeros([image.shape[0], image.shape[1]], dtype=complex)
    image_fft[:,:] = image[:,:,0]

    kernel_fft = np.zeros([image.shape[0], image.shape[1]], dtype=complex)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel_fft[i,j] = (kernel[i,j])


    image_fft = fft2(image_fft)
    kernel_fft = fft2(kernel_fft)

    res_fft = np.zeros([image.shape[0], image.shape[1]],dtype=complex)
    res_fft = image_fft * kernel_fft
    res_fft = ifft2(res_fft).real

    image_conv = np.zeros([res_fft.shape[0],res_fft.shape[1],3],dtype=int)
    for i in range(res_fft.shape[0]):
        for j in range(res_fft.shape[1]):
            image_conv[i,j,:] = res_fft[i,j]
    return image_conv

def normalize(image):
    return (256 * ((image-image.min()) / (1+image.max()-image.min()))).astype(np.uint8)


def korr(sig1, sig2):
    return convolution_fft(sig1, np.flip(sig2[:,:,0]))


kernel1 =  np.array([[7,32,16],
                    [8,-40,15],
                    [0,-5,1.5]])
kernel2 =  np.array([[1,0,0,0,0,0,1],
                     [1,0,0,0,0,0,1],
                     [1,0,0,0,0,0,1]])
kernel3 =  np.array([[0.33,0,0],
                     [0,0.33,0],
                     [0,0,0.33]])

image_conv = convolution_by_map(image_gray, kernel3)
show_image(image_conv)

image_conv = convolution_fft(image_gray, kernel3)
show_image(image_conv)


show_image(normalize(korr(image_noise, image_gray)))

show_image(normalize(korr(image_noise, image_noise)))

D = [[1,1],[2,2],[3,3]]
