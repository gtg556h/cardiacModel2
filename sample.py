import numpy as np
import pdb
import matplotlib.pyplot as plt

import oscillatorLib as ol




dt = 0.0001
maxTime = 4
staticRate = 0.8
leakRate = 0.2
randomStd = 0.1
peakEpsilon = 1
actionPotentialLength = 0.25
contractionDelay = 0.05
peakForce = 1
c0 = 0
sensitivityWindow = 1
peakCouplingRate = 1
sensitivityMean = 0.5
sensitivityStd = 0.1
title = "cardiacOscillator_1"


c1 = ol.cardiac(dt, maxTime, staticRate, leakRate, randomStd, peakEpsilon, actionPotentialLength, contractionDelay, peakForce, c0, sensitivityWindow, peakCouplingRate, sensitivityMean, sensitivityStd, title)


for i in range(1,c1.c.size):
    c1.stepTime(i,1)


plt.plot(c1.t, c1.c)
plt.show()
