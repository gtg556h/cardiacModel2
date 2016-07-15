import numpy as np
import numpy.random



i = 500

dtInt = 100
dt = 1/dtInt

std1 = 1
std2 = std1*np.sqrt(dt)

x1 = np.zeros(500)
x2 = np.zeros(500)

for i in range(0, 500):
    x1[i] = numpy.random.normal(0, std1)

    for j in range(0,dtInt):
        x2[i] = x2[i] + numpy.random.normal(0, std2)

    
