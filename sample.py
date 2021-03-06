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
staticRate = 0.8
leakRate = 0#0.2
randomStd = 0.2
peakEpsilon = 1
actionPotentialLength = 0.25
contractionDelay = 0.05
peakForce = 1
peakCouplingRate = 2
c0 = 0
sensitivityWinType = 1
sensitivityMean = 0.4
sensitivityStd = 0.1
cellTitle = "cardiacOscillator_1"


####################################
### SUBSTRATE PARAMETERS:

funcType = 'sinusoidal'
minStrain = 0
maxStrain = 1
phi0 = 1.5*np.pi


omega1 = 0.8 * 2*np.pi
omega2 = 0.9 * 2*np.pi


####################################
####################################
####################################

sensitivityWinParam = {'sensitivityWinType' : sensitivityWinType, 'sensitivityMean' : sensitivityMean, 'sensitivityStd' : sensitivityStd}

strainFunctionParameters = {'funcType' : funcType, 'minStrain' : minStrain, 'maxStrain' : maxStrain, 'phi0' : phi0}

cell = ol.cardiac(dt, maxTime, staticRate, leakRate, randomStd, peakEpsilon, actionPotentialLength, contractionDelay, peakForce, c0, peakCouplingRate, sensitivityWinParam, cellTitle)

sub1 = ol.substrate(dt, maxTime, strainFunctionParameters, omega1)

sub2 = ol.substrate(dt, maxTime, strainFunctionParameters, omega2)

#cell.simulatePerturbed(sub.epsilon)


uncoupled = ol.ap(cell, 0)
print("step")
coupled1 = ol.ap(cell, sub1.epsilon)
print("step")
coupled2 = ol.ap(cell, sub2.epsilon)
print("step")
#####################################
#####################################
#####################################

if 0:
    plt.plot(uncoupled.t, uncoupled.c, coupled.t, coupled.c)
    plt.show()






##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
########  POSTPROCESS DATA USING EXPERIMENTAL LIBS   #########
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################

experimentTitle = '20160605_substrate1'




maxStrain = .13

cellNaturalFreq = 1/np.mean(np.diff(uncoupled.ix))

#######################################
# Optional:  Use nominal substrate frequency (no phase data!)

useAvailableSubstrateEvents = 1  # Set to unity to use *data* for substrate, as opposed to measured frequency



cellEvents = [uncoupled.ix, coupled1.ix, coupled2.ix]
subEvents = [[], sub1.ix, sub2.ix]
title = ['uncoup', 'coup1', 'coup2']

voltage = np.zeros(len(cellEvents))
startTime = np.arange(len(cellEvents))
nominalSubFreq = np.zeros(len(cellEvents))
reactionTime = 0
subEventTimeShift = np.zeros(len(cellEvents))



params = {'cellEvents':cellEvents, 'subEvents':subEvents, 'voltage':voltage, 'startTime':startTime, 'dt':dt, 'useAvailableSubstrateEvents':useAvailableSubstrateEvents, 'nominalSubFreq':nominalSubFreq, 'reactionTime':reactionTime, 'subEventTimeShift':subEventTimeShift, 'maxStrain':maxStrain, 'cellNaturalFreq':cellNaturalFreq, 'title':title, 'experimentTitle':experimentTitle}



s1 = experimentLib.experiment(params)

u0 = s1.genUnstretchedMeasurement(0)
m1 = s1.genStretchedMeasurement(1)
m2 = s1.genStretchedMeasurement(2)


if 0:
    ol.animateCoupledUncoupled(sub1.epsilon, coupled1.c, uncoupled.c, DF=500)

#sub1.epsilon




print("cellFreq(u0) =", u0.cellFreq)
print()
print("subFreq(m1) =", m1.subFreq)
print("cellFreq(m1) =", m1.cellFreq)
print()
print("subFreq(m2) =", m2.subFreq)
print("cellFreq(m2) =", m2.cellFreq)

plt.plot(m1.t2, m1.dTheta, m2.t2, m2.dTheta)
plt.show()












