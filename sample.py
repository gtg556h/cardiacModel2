import numpy as np
import pdb
import matplotlib.pyplot as plt
import oscillatorLib as ol


######################################
######################################
######  SIMULATION PARAMETERS   ######
######################################
######################################

#####################################
### SYSTEM PARAMETERS:
dt = 0.0001
maxTime = 4


#####################################
### CELL PARAMETERS:
staticRate = 0.8
leakRate = 0.2
randomStd = 0.1
peakEpsilon = 1
actionPotentialLength = 0.25
contractionDelay = 0.05
peakForce = 1
peakCouplingRate = 1
c0 = 0
sensitivityWinType = 1
sensitivityMean = 0.5
sensitivityStd = 0.1
cellTitle = "cardiacOscillator_1"


####################################
### SUBSTRATE PARAMETERS:
omega = 2*np.pi
funcType = 'sinusoidal'
minStrain = 0
maxStrain = 1
phi0 = 1.5*np.pi

####################################
####################################
####################################

sensitivityWinParam = {'sensitivityWinType' : sensitivityWinType, 'sensitivityMean' : sensitivityMean, 'sensitivityStd' : sensitivityStd}

strainFunctionParameters = {'funcType' : funcType, 'minStrain' : minStrain, 'maxStrain' : maxStrain, 'phi0' : phi0}

cell = ol.cardiac(dt, maxTime, staticRate, leakRate, randomStd, peakEpsilon, actionPotentialLength, contractionDelay, peakForce, c0, peakCouplingRate, sensitivityWinParam, cellTitle)

sub = ol.substrate(dt, maxTime, strainFunctionParameters, omega)
cell.simulatePerturbed(sub.epsilon)


#####################################
#####################################
#####################################

plt.plot(cell.t, cell.c, sub.t, sub.epsilon)
plt.show()






