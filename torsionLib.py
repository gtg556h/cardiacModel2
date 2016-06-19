from __future__ import division
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import numpy.linalg
import matplotlib.animation as animation
import random
import pdb


def genWaveform(force, duration, dt):
    tShort = np.arange(0, duration, dt)
    f = force * np.sin(tShort * np.pi / duration)

    return f 


class torsion(object):

    def __init__(self, params):

        self.dt = params['dt']
        self.tMax = params['tMax']

        self.k = params['k']
        self.force = params['force']
        self.duration = params['duration']
        self.sign = params['sign']

        self.kRand = params['kRand']
        self.kConst = params['kConst']
        self.kCoup = params['kCoup']
        self.kLeak = params['kLeak']
        self.c0 = params['c0']

        self.fFunc = genWaveform(self.force, self.duration, self.dt)
        self.t = np.arange(0, self.tMax, self.dt)
        self.nt = self.t.shape[0]

        self.c = np.zeros([self.nt, 2])
        self.c[0,:] = self.c0
        self.state = np.zeros([self.nt, 2])
        self.cRand = np.zeros([self.nt, 2])
        self.cConst = np.zeros([self.nt, 2])
        self.cCoup = np.zeros([self.nt, 2])
        self.cLeak = np.zeros([self.nt, 2])
        self.f = np.zeros([self.nt, 2])

        self.nActuators = 2

        self.x = np.zeros(self.nt)

    def stepC(self,i):
        thresh = self.fFunc.shape[0] - 1

        for jj in range(0,self.nActuators):
            if self.state[i-1,jj] > 0:
                self.state[i,jj] = self.state[i-1,jj] + 1
                self.f[i,jj] = self.fFunc[self.state[i-1,jj]]
                self.cCoup[i,jj] = np.max([0, self.sign[jj] * self.x[i-1]*self.kCoup[jj] * self.dt])

                if self.state[i,jj] > thresh:
                    self.state[i,jj] = 0
            else:
                self.cCoup[i,jj] = np.max([0, self.sign[jj] * self.x[i-1] * self.kCoup[jj] * self.dt])
                self.cRand[i,jj] = self.dt * self.kRand[jj] * random.random()
                self.cConst[i,jj] = self.dt * self.kConst[jj]
                self.cLeak[i,jj] = self.dt * self.kLeak[jj] * self.c[i-1,jj] 

                self.c[i,jj] = self.c[i-1,jj] + self.cCoup[i,jj] + self.cConst[i,jj] + self.cRand[i,jj] - self.cLeak[i,jj] 

                if self.c[i,jj] > 1:
                    self.state[i,jj] = 1
                    self.c[i,jj] = 0


    def simulate(self):
        for i in range(1,self.nt):
            self.stepC(i)
            for jj in range(0,self.nActuators):
                self.x[i] = self.x[i] - self.sign[jj]*self.f[i,jj]*self.k



    def plotDisp(self, DF=10, plotFrac=1):
        if 0:
            plt.plot(self.c)
            plt.show()

        if 1:
            nt = self.nt
            dx=.1

            nFrames = np.int(np.floor(nt/DF*plotFrac))
            
            fig = plt.figure(figsize=[8,6])
            ax = plt.axes()

            cellRadius = 20
            ax.set_xlim([-1.2*np.max(self.x)-cellRadius,1.2*np.max(self.x)+cellRadius])
            ax.set_ylim([-1.2,1.2])

            line1, = ax.plot([], [], lw=4)
            line2, = ax.plot([], [], lw=4)
            line3, = ax.plot([], [], lw=2)

            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                line3.set_data([], [])
                return line1, line2, line3

            def animate(i):
                x1 = np.arange(-self.x[DF*i]-cellRadius,self.x[DF*i]+cellRadius,dx)
                x2 = np.arange(self.x[DF*i]-cellRadius, -self.x[DF*i]+cellRadius, dx)
                x3 = np.arange(-cellRadius, cellRadius, dx)

                y1 = np.ones(x1.shape)
                y2 = -1*np.ones(x2.shape)
                y3 = np.zeros(x3.shape)

                line1.set_data(x1,y1)
                line2.set_data(x2,y2)
                line3.set_data(x3,y3)
                return line1, line2, line3,

            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=50, blit=True, repeat=False)
            #pdb.set_trace()

            plt.show()
        
    def saveVid3(self, filename, DF=10, plotFrac=0.10):

        nt = self.nt

        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='output')
        writer = FFMpegWriter(fps=30, metadata=metadata)

        fig = plt.figure(figsize=[8,8])
        ax = plt.axes()
        ax.set_axis_off()

        tRange = 3*self.duration
        tInd = np.round(tRange/self.dt)

        cellRadius = 20
        dx = 0.1
        ax.set_xlim([-1.2*np.max(self.x)-cellRadius, 1.2*np.max(self.x)+cellRadius])
        ax.set_ylim([-1.2, 1.2])

        line1, = ax.plot([], [], lw=4)
        line2, = ax.plot([], [], lw=4)
        line3, = ax.plot([], [], lw=2)
        line4, = ax.plot([], [], lw=2)
        line5, = ax.plot([], [], lw=2)

        line1.set_color('blue')
        line2.set_color('blue')
        line3.set_color('black')
        line4.set_color('black')
        line5.set_color('black')

        nFrames = np.int(np.floor(nt/DF*plotFrac))

        #with writer.saving(fig, "modelSolution.mp4", nFrames):
        with writer.saving(fig, filename, nFrames):
            for i in range(nFrames):
                print(i/nFrames)

                ii = i*DF

                if self.t[ii] > tRange:
                    if self.t[ii] < np.max(self.t) - tRange:
                        y6 = self.c[ii-tInd:ii+tInd,0]
                        y7 = self.c[ii-tInd:ii+tInd,1]
                    else:
                        y6 = self.c[ii-tInd:nt,0]
                        y7 = self.c[ii-tInd:nt,1]

                else:
                    y6 = self.c[0:ii+tInd,0]
                    y7 = self.c[0:ii+tInd,0] 

                y6 = y6 + 1.2
                y7 = y7 - 2.2

                x1 = np.arange(-self.x[ii]-cellRadius, self.x[ii]+cellRadius,dx)
                x2 = np.arange(self.x[ii]-cellRadius, -self.x[ii]+cellRadius, dx)
                y1 = np.ones(x1.shape[0])
                y2 = -1*np.ones(x2.shape[0])

                if self.state[ii,0] != 0:
                    line1.set_color('red')
                else:
                    line1.set_color('blue')

                if self.state[ii,1] != 0:
                    line2.set_color('red')
                else:
                    line2.set_color('blue')

                x3 = np.arange(-cellRadius, cellRadius, dx)
                y3 = np.zeros(x3.shape[0])

                x4 = np.array([-self.x[ii]-cellRadius, self.x[ii]-cellRadius])
                y4 = np.array([1,-1])

                x5 = np.array([self.x[ii]+cellRadius, -self.x[ii]+cellRadius])
                y5 = np.array([1,-1])

                line1.set_data(x1,y1)
                line2.set_data(x2,y2)
                line3.set_data(x3,y3)
                line4.set_data(x4,y4)
                line5.set_data(x5,y5)

                writer.grab_frame()

    def saveVid(self, filename, DF=10, plotFrac=0.10):
        # saveVid2: adds [Ca] plot above and below figure

        nt = self.nt

        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='output')
        writer = FFMpegWriter(fps=30, metadata=metadata)

        fig = plt.figure(figsize=[8,8])
        ax = plt.axes()
        ax.set_axis_off()

        cellRadius = 20
        dx = 0.1
        ax.set_xlim([-1.2*np.max(self.x)-cellRadius, 1.2*np.max(self.x)+cellRadius])
        ax.set_ylim([-1.2, 1.2])

        line1, = ax.plot([], [], lw=4)
        line2, = ax.plot([], [], lw=4)
        line3, = ax.plot([], [], lw=2)
        line4, = ax.plot([], [], lw=2)
        line5, = ax.plot([], [], lw=2)

        line1.set_color('blue')
        line2.set_color('blue')
        line3.set_color('black')
        line4.set_color('black')
        line5.set_color('black')

        nFrames = np.int(np.floor(nt/DF*plotFrac))

        #with writer.saving(fig, "modelSolution.mp4", nFrames):
        with writer.saving(fig, filename, nFrames):
            for i in range(nFrames):
                print(i/nFrames)

                ii = i*DF

                x1 = np.arange(-self.x[ii]-cellRadius, self.x[ii]+cellRadius,dx)
                x2 = np.arange(self.x[ii]-cellRadius, -self.x[ii]+cellRadius, dx)
                y1 = np.ones(x1.shape[0])
                y2 = -1*np.ones(x2.shape[0])

                if self.state[ii,0] != 0:
                    line1.set_color('red')
                else:
                    line1.set_color('blue')

                if self.state[ii,1] != 0:
                    line2.set_color('red')
                else:
                    line2.set_color('blue')

                x3 = np.arange(-cellRadius, cellRadius, dx)
                y3 = np.zeros(x3.shape[0])

                x4 = np.array([-self.x[ii]-cellRadius, self.x[ii]-cellRadius])
                y4 = np.array([1,-1])

                x5 = np.array([self.x[ii]+cellRadius, -self.x[ii]+cellRadius])
                y5 = np.array([1,-1])

                line1.set_data(x1,y1)
                line2.set_data(x2,y2)
                line3.set_data(x3,y3)
                line4.set_data(x4,y4)
                line5.set_data(x5,y5)

                writer.grab_frame()


                

    def saveModel(self, saveFilename):
        np.savez(saveFilename, c=self.c, cCoup = self.cCoup, cRand = self.cRand, cConst = self.cConst,  x = self.x, cLeak = self.cLeak, state = self.state, f = self.f, kLeak = self.kLeak, kCoup = self.kCoup, c0 = self.c0, kConst = self.kConst, t = self.t, kRand = self.kRand)



    def plotScript(self):
        c3=1
