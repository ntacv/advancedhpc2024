
from array import *
import numpy as np
import matplotlib.pyplot as plt

import time


matrix_blur = [[0,1,0],[1,2,1],[0,1,0]]
print(type(matrix_blur))
print(matrix_blur)
print( [matrix_blur[i][0:2] for i in range(0,2)] )



imageWidth = 1000
imageHeight = 1000
blockSize = 64
maxBlockSize = 80

"""
x_image
y_image

for i in (-3,4)
for j in (-3,4)

cuda.shared.array(matrix_blur)
"""


matrix_image = np.zeros((imageHeight, imageWidth, 3)) #np.random.randint(0, 255, (imageHeight, imageWidth, 3), dtype=np.uint8)

for i in range(imageHeight): 
  for j in range(imageWidth): 
    for k in range(len(matrix_image[0][0])): 
      matrix_image[i][j][k] = int(str(i)+str(j)) #+str(k)


def matrix_product (max1, max2): 
  max_out = np.zeros(shape=(len(max1), len(max1[0])))
  for i in range(len(max1)): 
    for j in range(len(max1[i])):
       max_out[i][j] = max1[i][j] * max2[i][j]
  return max_out

def matrix_mult (max1, max2): 
  sum = 0
  for i in range(len(max1)): 
    for j in range(len(max1[i])):
       sum += max1[i][j] * max2[i][j]
  return sum


matrix_blurred_image = np.zeros(shape=(len(matrix_image),len(matrix_image[0])))

matrix_test = matrix_blur

print(matrix_image[0:2][0:2])

print("### LOOP")
for i in range(imageHeight+len(matrix_blur)): 
  for j in range(imageWidth+len(matrix_blur)): 
      if i in range(len(matrix_blur)//2, imageHeight-len(matrix_blur)//2) and j in range(len(matrix_blur[0])//2, imageWidth-len(matrix_blur[0])//2): 
        #for k in range (len(matrix_image[0][0])): 
        blur_height = len(matrix_blur)//2
        blur_width = len(matrix_blur[0])//2
        print(blur_height, blur_width)
        
        matrix_image_tmp = matrix_image[(i-len(matrix_blur)//2):(i+len(matrix_blur)//2)][j-len(matrix_blur[0])//2:j+len(matrix_blur[0])//2]
        print(matrix_image_tmp)
        
        matrix_mult(matrix_image_tmp, matrix_blur)
