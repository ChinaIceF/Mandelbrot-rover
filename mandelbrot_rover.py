# numba_cuda_test.py
#import cv2 as cv
import numpy as np
from numpy import *
from numba import cuda
from PIL import Image
import time
from matplotlib import *

"""
@cuda.jit
def gpu():
  
  print('blockId_x:', cuda.blockIdx.x, 'blockId_y:', cuda.blockIdx.y, )
  print("gh")
  

if __name__ == '__main__':
  gpu[2,4]()
  print("Done")
  input()
"""
@cuda.jit(device=True)
def mandel(x, y, max_iters):
  """
  Given the real and imaginary parts of a complex number,
  determine if it is a candidate for membership in the Mandelbrot
  set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = (z * z + c)# complex(z.imag, z.real)
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel(real, imag, iters)

def gpu_get_pic(pos_R, pos_I, delta, pixel_rate, times = 2560):
    #pos_R, pos_I, delta, pixel_rate = (0.2537269133080432, 0.000365995381749671135, 0.0000000001, 4)
    print(pos_R, pos_I, delta, pixel_rate)

    gimage = np.zeros((1024*pixel_rate, 1024*pixel_rate), dtype = np.uint8)
    blockdim = (64, 8)
    griddim = (128,64)

    start = time.time()
    d_image = cuda.to_device(gimage)
    mandel_kernel[griddim, blockdim](pos_R - delta, pos_R + delta, pos_I - delta, pos_I + delta, d_image, 2560) 
    #mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 256) 
    output = d_image.copy_to_host()
    dt = time.time() - start
    
    return output
    #print("Mandelbrot created on GPU in %f s" % dt)

    #print(output.max())
    #Image.fromarray(numpy.uint8(output*1.5)).save("result_mang.png")

#np.save("a.png",gimage)
#cv.imshow(gimage)
#On a server with an NVIDIA Tesla P100 GPU and an Intel Xeon E5-2698 v3 CPU, this CUDA Python Mandelbrot code runs nearly 1700 times faster than the pure Python version. 1700x may seem an unrealistic speedup, but keep in mind that we are comparing compiled, parallel, GPU-accelerated Python code to interpreted, single-threaded Py

if __name__ == '__main__':
    data_list = []
    for i in range(49):
        rate = numpy.float64(numpy.power(0.5, i))
        _r = numpy.float64(0.29341989288727477)
        _i = numpy.float64(0.481365430202407)
        pic = gpu_get_pic(_r, _i, rate, 2, 2560)
        #data_list.append(pic)
        print(i)
    
    #for i in range(100):
        Image.fromarray(pic).save("./voyage/{}.png".format(i))
