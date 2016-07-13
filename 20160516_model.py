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
randomStd = 0.2
peakEpsilon = 1
epsilon_floor = 0.8
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

#d_omega = 0.1 * omega0
#omega1 = omega0 + 1*d_omega
#omega2 = omega0 + 2*d_omega
#omega3 = omega0 + 3*d_omega
#omega4 = omega0 + 4*d_omega
#omega5 = omega0 + 5*d_omega
#omega6 = omega0 + 6*d_omega

omega1 = 1.14 * omega0
omega2 = 1.26 * omega0
omega3 = 1.56 * omega0
omega4 = 1.57 * omega0
omega5 = 1.72 * omega0
omega6 = 1.98 * omega0


####################################
####################################
####################################

sensitivityWinParam = {'sensitivityWinType' : sensitivityWinType, 'sensitivityMean' : sensitivityMean, 'sensitivityStd' : sensitivityStd}

strainFunctionParameters = {'funcType' : funcType, 'minStrain' : minStrain, 'maxStrain' : maxStrain, 'phi0' : phi0}

cell = ol.cardiac(dt, maxTime, staticRate, leakRate, randomStd, peakEpsilon, epsilon_floor, actionPotentialLength, contractionDelay, peakForce, c0, peakCouplingRate, sensitivityWinParam, cellTitle)

sub1 = ol.substrate(dt, maxTime, strainFunctionParameters, omega1)
sub2 = ol.substrate(dt, maxTime, strainFunctionParameters, omega2)
sub3 = ol.substrate(dt, maxTime, strainFunctionParameters, omega3)
sub4 = ol.substrate(dt, maxTime, strainFunctionParameters, omega4)
sub5 = ol.substrate(dt, maxTime, strainFunctionParameters, omega5)
sub6 = ol.substrate(dt, maxTime, strainFunctionParameters, omega6)

#cell.simulatePerturbed(sub.epsilon)


uncoupled = ol.ap(cell, 0)
print("step")
coupled1 = ol.ap(cell, sub1.epsilon)
print("step")
coupled2 = ol.ap(cell, sub2.epsilon)
print("step")
coupled3 = ol.ap(cell, sub3.epsilon)
print("step")
coupled4 = ol.ap(cell, sub4.epsilon)
print("step")
coupled5 = ol.ap(cell, sub5.epsilon)
print("step")
coupled6 = ol.ap(cell, sub6.epsilon)

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



cellEvents = [uncoupled.ix, coupled1.ix, coupled2.ix, coupled3.ix, coupled4.ix, coupled5.ix, coupled6.ix]
subEvents = [[], sub1.ix, sub2.ix, sub3.ix, sub4.ix, sub5.ix, sub6.ix]
title = ['uncoup', 'coup1', 'coup2', 'coup3', 'coup4', 'coup5', 'coup6']

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
m3 = s1.genStretchedMeasurement(3)
m4 = s1.genStretchedMeasurement(4)
m5 = s1.genStretchedMeasurement(5)
m6 = s1.genStretchedMeasurement(6)




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
print()
print("subFreq(m3) =", m3.subFreq)
print("cellFreq(m3) =", m3.cellFreq)
print()
print("subFreq(m4) =", m4.subFreq)
print("cellFreq(m4) =", m4.cellFreq)
print()
print("subFreq(m5) =", m5.subFreq)
print("cellFreq(m5) =", m5.cellFreq)
print()
print("subFreq(m6) =", m6.subFreq)
print("cellFreq(m6) =", m6.cellFreq) # 

#plt.plot(m1.t2, m1.dTheta, m2.t2, m2.dTheta, m3.t2, m3.dTheta, m4.t2, m4.dTheta, m5.t2, m5.dTheta)
#plt.show()


measurementList = [m1, m2, m3, m4, m5, m6]


nRows = 2
nColumns = 3
figsize = (14,6)
top=0.93
bottom=0.1
left=0.07
right=0.92
hspace=0.32   # vertical spacing between rows
wspace=0.22
hist_maxProbability = 1

s1.plotHistograms(measurementList, nRows, nColumns, figsize, top, bottom, left, right, hspace, wspace, hist_maxProbability)

####################################################

hist_maxProbability = 1.1
nBins = 16

#s3.stretchedTauHistogram(measurementList, nRows, nColumns, figsize, top, bottom, left, right, hspace, wspace, hist_maxProbability, nBins)











