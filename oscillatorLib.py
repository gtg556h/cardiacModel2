import numpy as np
import pdb
import matplotlib.pyplot as plt
import numpy.random
import matplotlib.animation as animation

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

        self.c0 = c0
        self.contractionDelay = contractionDelay
        self.actionPotentialLength = actionPotentialLength

        super().__init__(dt, title, maxTime)
        self.genSensitivityWindow(sensitivityWinParam)

                


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

    def stepTime(self):  # Make this a lambda func

        return lambda c, epsilon: (1 - self.k_leak) * c + self.k_constant + numpy.random.normal(0, self.k_random_std) + self.sensitivity(c) * epsilon * self.k_coup

    #############################################################
    


#################################################################
#################################################################
#################################################################

class ap(object):
    # ap for predetermined coupling function.  for cardiac-cardiac coupling, need
    # to modify.

    def __init__(self, cell, epsilon=0):
        # cell is a cardiac

        self.epsilon = epsilon
        if type(self.epsilon)==int:
            self.epsilon = np.ones_like(cell.t) * self.epsilon

        self.contractionDelay = cell.contractionDelay
        self.actionPotentialLength = cell.actionPotentialLength
        self.dt = cell.dt
        self.t = cell.t
        self.c = np.zeros_like(self.t)
        self.stepTime = cell.stepTime()
        # now self.stepTime used to march c:
        # c[i+1] = self.stepTime(c[i], epsilon[i])

        self.i = 0
        self.trig = []
        self.ix = []
        self.simulate()

        self.ix = np.asarray(self.ix)


    def simulate(self):
        while self.t[self.i] < self.t[-1]:
            self.i += 1
            self.c[self.i] = self.stepTime(self.c[self.i-1], self.epsilon[self.i-1])

            if self.c[self.i] > 1:
                self.trig.append(self.t[self.i])
                self.triggerAP()


    def triggerAP(self):

        if self.t[-1] - self.t[self.i] > self.contractionDelay:
            self.ix.append(self.trig[-1] + self.contractionDelay)

        if self.t[-1] - self.t[self.i] > self.actionPotentialLength:
            self.c[self.i : self.i + int(self.actionPotentialLength/self.dt)] = 2
            self.i += int(self.actionPotentialLength/self.dt) 
        else:
            self.c[self.i:] = 2
            self.i = self.t.size-1

        


            
#################################################################
#################################################################
#################################################################

class substrate(oscillator):

    def __init__(self, dt, maxTime, functionParameters, omega=2*np.pi, title="substrateOscillator"):
        
        self.omega = omega
        super().__init__(dt, title, maxTime)

        self.ix = []
        self.genFunc(functionParameters)

        self.findEvents()
        

    ######################################################

    def genFunc(self, functionParameters):

        if functionParameters['funcType'] == 'sinusoidal':

            minStrain = functionParameters['minStrain']
            maxStrain = functionParameters['maxStrain']
            phi0 = functionParameters['phi0']
            
            self.epsilon = 0.5*(maxStrain-minStrain) * (np.sin(self.omega*self.t + phi0) + 1)

            #for i in range(0,self.t.size-1):

    def findEvents(self):

        de = np.diff(self.epsilon)
        for i in range(0, de.size-1):
            if de[i] < 0 and de[i+1] >=0:
                print(i)
                self.ix.append(self.t[i])
        self.ix = np.asarray(self.ix)
    


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

    




##########################################################
##########################################################
##########################################################

# Model animations:


def animateCoupledUncoupled(epsilon, c, uc, DF=10, plotFrac=1):
    dx=.001
    nFrames = np.int(np.floor(epsilon.size/DF*plotFrac))
    fig = plt.figure(figsize=[8,6])
    ax = plt.axes()
    lw=12

    ax.set_xlim([-1.2*np.max(epsilon), 1.2*np.max(epsilon)])
    ax.set_ylim([-1.2,1.2])
    
    line1, = ax.plot([], [], lw=lw)
    line1.set_color("cyan")
    line2, = ax.plot([], [], lw=lw)
    line3, = ax.plot([], [], lw=lw)
    line3.set_color("cyan")
    line4, = ax.plot([], [], lw=lw)


    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])

        return line1, line2, line3, line4

    def animate(i):

        x1 = np.arange(-0.5*epsilon[DF*i]-0.5, 0.5*epsilon[DF*i]+0.5, dx)
        x2 = np.arange(-0.25*epsilon[DF*i]-0.25, 0.25*epsilon[DF*i]+0.25, dx)

        x3 = np.arange(-.5, .5, dx)
        x4 = np.arange(-.25, .25, dx)
        

        y1 = np.zeros(x1.shape)
        y2 = np.zeros(x2.shape) + .1
        y3 = np.zeros(x3.shape) - 0.4
        y4 = np.zeros(x4.shape) - 0.3

        if c[DF*i] == 2:
            line2.set_c("red")
        else:
            line2.set_c("blue")

        if uc[DF*i] == 2:
            line4.set_c("red")
        else:
            line4.set_c("blue")
        

        line1.set_data(x1,y1)
        line2.set_data(x2,y2)
        line3.set_data(x3,y3)
        line4.set_data(x4,y4)

        return line1, line2, line3, line4

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=50, blit=True, repeat=False)
    #pdb.set_trace()

    plt.show()
