import numba
import numba.cuda as cuda

import numpy as np
import matplotlib.pyplot as plt


imageWidth = 100
imageHeight = 100
blockSize = 64


hostData = np.random.randint(0, 255, (imageHeight, imageWidth, 3), dtype=np.uint8)

print(cuda.threadIdx.x)

@cuda.jit

#kernelName[numBlock, blockSize](args)
def grayscale(src, out): 
  srcSize = src.shape[1]
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  x = tidx % srcSize
  y = tidx // srcSize

  if x < srcSize and y < srcSize: 
    gray = np.uint8 ((src[x,y,0]+src[x,y, 1]+ src[x,y, 2]) / 3)
    out[x,y, 0] = out[x,y, 1] = out[x,y, 2] = gray

devData = cuda.to_device(hostData)
#devData = cuda.device_array

devOutput = cuda.device_array((imageHeight, imageWidth, 3), np.uint8)


pixelCount = imageWidth * imageHeight
gridSize = int(pixelCount / blockSize)


grayscale[gridSize, blockSize](devData, devOutput) #

hostOutput = devOutput.copy_to_host()

plt.imshow(hostOutput)
plt.show()
