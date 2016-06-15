import numpy as np
import pdb
import matplotlib.pyplot as plt
import numpy.random

###########################################################
###########################################################
###########################################################

# TODO
# Write in action potential
# Write in event extraction
# Couple with experiment lib
# Shift experiment to 2pi phase scale


###########################################################
###########################################################
###########################################################


class oscillator(object):
    # Generic attributes applicable to cells, substrates, etc...

    def __init__(self, dt, title, maxTime):
        self.dt = dt
        self.title = title
        # Preprocessing:
        self.genTime(maxTime)


    ######################################################

    def genTime(self, maxTime):
        self.t = np.arange(0, maxTime, self.dt)

    ######################################################
        
    def genIndices(self, events):
        ix = np.zeros(np.size(events))
        for i in range(0, np.size(events)):
            ix[i] = int(np.argmin(np.abs(events[i] - self.t)))

        return ix.astype(int)

    ######################################################
    
    def phaseGen(self,ix,t):
        phase = np.zeros(t.shape)
        ixDiff = np.diff(ix,n=1,axis=0)

        for ii in range(0,ix.shape[0]-1):
            #phase[ix[ii,0],0] = 0
            for jj in range(0,int(ixDiff[ii])+1):
                phase[int(ix[ii])+jj] = float(jj)/float(ixDiff[ii])

        return phase #, ixDiff


##############################################################
##############################################################
##############################################################


class cardiac(oscillator):

    def __init__(self, dt, maxTime, staticRate=0.8, leakRate = 0.1, randomStd=0.1, peakEpsilon=1, actionPotentialLength=0.25, contractionDelay=0.05, peakForce=1, c0 = 0, peakCouplingRate=0.2, sensitivityWinParam = {'sensitivityWinType' : 0}, title="cardiacOscillator"):
        
        self.cellEvents = []
        # k_constant computed such that, all other factors aside, we accumulate staticRate of our accumulation
        # variable every unit time
        self.k_constant = staticRate*dt

        # For random component:  We want the random step to be a zero mean normal distribution
        # with a standard deviation after a unit time equivalent to randomStd
        # Considering Gaussian random walks, where each time step has standard deviation
        # k_random_std, after n timesteps, we have a normal distribution with standard
        # deviation sqrt(n)*k_random_std
        self.k_random_std = randomStd * np.sqrt(dt) * dt

        # k_leak, the leak rate per timestep, is computed such that after n timesteps in a unit time
        # we get a reduction in integrating variable by 1-leakRate:
        self.k_leak = 1-(1-leakRate)**(dt)

        # k_coup:  The coupling strength.  Scaled based on desired peak coupling
        self.k_coup = dt * peakCouplingRate / peakEpsilon

        super().__init__(dt, title, maxTime)
        self.genSensitivityWindow(sensitivityWinParam)

        self.c = np.zeros_like(self.t)
        self.c[0] = c0
        self.c_unperturbed = np.zeros_like(self.t)
        self.c_unperturbed[0] = c0
        self.simulateUnperturbed()


    #############################################################

    def genSensitivityWindow(self, sensitivityWinParam):
        # Set sensitivity windowing function:
        sensitivityWinType = sensitivityWinParam['sensitivityWinType']

        # 0 implies constant sensitivity of unity:
        if sensitivityWinType == 0:
            self.sensitivity = lambda c: 1

        # 1 implies a gaussian sensitivity
        if sensitivityWinType == 1:
            sensitivityMean = sensitivityWinParam['sensitivityMean']
            sensitivityStd = sensitivityWinParam['sensitivityStd']
            self.sensitivity = lambda c: np.exp(-(c-sensitivityMean)**2/(2*sensitivityStd**2))

        
    #############################################################

    def simulateUncoupled(self):

        self.u = ap()

        
        for i in range(1, self.c.size):
            self.stepTime(i, self.c_unperturbed, 0)


    #############################################################
            
    def simulateCoupled(self, epsilon):
        for i in range(1, self.c.size):
            self.stepTime(i, self.c, epsilon[i])

            
    #############################################################

    def stepTime(self, i, c, epsilon):  # Make this a lambda func

        c[i] = (1 - self.k_leak) * c[i-1] + self.k_constant + numpy.random.normal(0, self.k_random_std) + self.sensitivity(c[i-1]) * epsilon * self.k_coup
        if c[i] > 1:
            self.triggerAP()
            c[i] = 0


    #############################################################
    
    def triggerAP(self, i, c):



#################################################################
#################################################################
#################################################################

class ap(object):
    # ap for predetermined coupling function.  for cardiac-cardiac coupling, need
    # to modify.

    def __init__(self, card, epsilon=np.zeros_like(card.t)):
        # cell is a cardiac

    self.stepTime = card.stepTime


    def simulate(self):
        for i in range(1, self.t.size):
            self.stepTime(i, self.c, epsilon[i])
        
#################################################################
#################################################################
#################################################################

class substrate(oscillator):

    def __init__(self, dt, maxTime, functionParameters, omega=2*np.pi, title="substrateOscillator"):
        
        self.omega = omega
        super().__init__(dt, title, maxTime)

        self.genFunc(functionParameters)
        

    ######################################################

    def genFunc(self, functionParameters):

        if functionParameters['funcType'] == 'sinusoidal':
            minStrain = functionParameters['minStrain']
            maxStrain = functionParameters['maxStrain']
            phi0 = functionParameters['phi0']
            
            self.epsilon = 0.5*(maxStrain-minStrain) * (np.sin(self.omega*self.t + phi0) + 1)
    


#######################################################
#######################################################
#######################################################

# Orphaned functions:
def relativePhase(self, subTheta, subIx):
    if subTheta.size > self.cellTheta.size:
        subTheta = subTheta[0:self.cellTheta.size]
    elif subTheta.size < self.cellTheta.size:
        self.cellTheta = self.cellTheta[0:subTheta.size]
    dTheta = np.mod(self.cellTheta - subTheta, 1)
    minIndex = int(np.max([np.min(self.cellIx), np.min(subIx)]))
    maxIndex = int(np.min([np.max(self.cellIx), np.max(subIx)]))
    dTheta2 = dTheta[minIndex:maxIndex]
    t2 = self.t[minIndex:maxIndex]

    







