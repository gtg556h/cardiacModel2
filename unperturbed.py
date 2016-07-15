import numpy as np
import pdb
import matplotlib.pyplot as plt
import oscillatorLib as ol
import experimentLib
import matplotlib
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter



######################################
######################################
######  SIMULATION PARAMETERS   ######
######################################
######################################

#####################################
### SYSTEM PARAMETERS:
dt = 0.0001
maxTime = 45


#####################################
### CELL PARAMETERS:
staticRate = 1/1.78  # Computed as 1/((1/.429) - 0.55), where .429 is the frequency of cells from 20160516
leakRate = 0#0.2

# Note: std of unstretched contractile events was 0.055 (averaged over three trials)
# random Std was chosen to generate a similar value:
randomStd = .0245
peakEpsilon = 1
epsilon_floor = 0.8  # Provides nonlinearity...
actionPotentialLength = 0.5  # measured from contractions on 20160516
contractionDelay = 0.005  # pretty random i suppose
peakForce = 1
peakCouplingRate = 10
c0 = 0
sensitivityWinType = 1
sensitivityMean = 0.6
sensitivityStd = 0.1
cellTitle = "cardiacOscillator_1"


####################################
### SUBSTRATE PARAMETERS:

funcType = 'sinusoidal'
minStrain = 0
maxStrain = 1
phi0 = 1.5*np.pi


omega0 = 2*np.pi / (actionPotentialLength + 1/staticRate)  # Note, this is natural contractile rate of cell
omega1 = 1.14 * omega0


####################################
####################################
####################################

sensitivityWinParam = {'sensitivityWinType' : sensitivityWinType, 'sensitivityMean' : sensitivityMean, 'sensitivityStd' : sensitivityStd}

strainFunctionParameters = {'funcType' : funcType, 'minStrain' : minStrain, 'maxStrain' : maxStrain, 'phi0' : phi0}

cell = ol.cardiac(dt, maxTime, staticRate, leakRate, randomStd, peakEpsilon, epsilon_floor, actionPotentialLength, contractionDelay, peakForce, c0, peakCouplingRate, sensitivityWinParam, cellTitle)

sub1 = ol.substrate(dt, maxTime, strainFunctionParameters, omega1)

uncoupled0 = ol.ap(cell, 0)
print("step")
#coupled1 = ol.ap(cell, sub1.epsilon)
#print("step")


#####################################
#####################################
#####################################



if 0
    plt.plot(x)







