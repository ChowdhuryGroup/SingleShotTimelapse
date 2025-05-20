# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import math

ro = 1.293 #kg/m^3

# %%
# Class to handle loading and fitting of positions
class ShockwaveData:
    def __init__(self, filepath, PixPerMicron=4.04, name=None):
        self.df = pd.read_csv(filepath, sep="\t")
        self.time = self.df["time"]
        self.positions = self.df.iloc[:, 1:] / PixPerMicron
        self.numberChannels = self.positions.shape[1]
        self.probeTimes = [3, 2, 1, 0]  # ns times of when probe channels arrive
        self.name = name if name is not None else filepath

    def imageToImageVelocity(self, channel):
        # calculates difference of position on "channel" from image 1 to image 3 divided by the time between image 1 and 3
        time_diff = self.time.shift(-1) - self.time.shift(1)
        position_diff = self.positions[f"channel{channel}X coord"].shift(
            -1
        ) - self.positions[f"channel{channel}X coord"].shift(1)
        velocity = position_diff / time_diff
        return velocity.iloc[1:-1]

    def positionPolynomial(self, channel, degree=1): #this was degree three
        coefficients = np.polyfit(
            self.getTimes(), self.getChannelPositions(channel), degree
        )
        return np.poly1d(coefficients)

    def velocityPolynomial(self, channel):
        return np.polyder(self.positionPolynomial(channel))

    def singleImageVelocities(self, first_channel=1, last_channel=4):
        velocity_in_single_image = (
            self.getChannelPositions(first_channel)
            - self.getChannelPositions(last_channel)
        ) / (self.probeTimes[last_channel - 1] - self.probeTimes[first_channel - 1])
        return velocity_in_single_image

    def getChannelPositions(self, channel):
        return self.positions[f"channel{channel}X coord"]

    def getTimes(self,channel=1):
        return self.time+self.probeTimes[channel]

    def getName(self):
        return self.name

    def getNumberOfChannels(self):
        return self.numberChannels


# %%
# Loading in Data Files
file1, intensity1 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ta_190e15.tsv", "Ta 1.90e15W/cm^2"
file2, intensity2 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ta_283e15.tsv", "Ta 2.83e15W/cm^2"
file3, intensity3 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ta_551e15.tsv", "Ta 5.51e15W/cm^2"
file4, intensity4 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Plastic_256e15.tsv", "Plastic 2.56e15W/cm^2"
file5, intensity5 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Plastic_653e15.tsv", "Plastic 6.53e15W/cm^2"
file6, intensity6 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Plastic_692e15.tsv", "Plastic 6.92e15W/cm^2"
file7, intensity7 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_Ta_102e9.tsv", "Ng Ta 1.02e9W/cm^2"
file8, intensity8 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_Ta_169e9.tsv", "Ng Ta 1.69e9W/cm^2"
file9, intensity9 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_plastic_102e9.tsv", "Ng Plastic 1.02e9W/cm^2"
file10, intensity10 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_plastic_159e9.tsv", "Ng Plastic 1.59e9W/cm^2"


Ta_190e15 = ShockwaveData(file1, name=intensity1)
Ta_283e15 = ShockwaveData(file2, name=intensity2)
Ta_551e15 = ShockwaveData(file3, name=intensity3)
Plastic_256e15 = ShockwaveData(file4, name=intensity4)
Plastic_653e15 = ShockwaveData(file5, name=intensity5)
Plastic_692e15 = ShockwaveData(file6, name=intensity6)
Ng_Ta_102e9 = ShockwaveData(file7, name=intensity7,PixPerMicron=4.5429)
Ng_Ta_169e9 = ShockwaveData(file8, name=intensity8,PixPerMicron=4.5429)
Ng_plastic_102e9 = ShockwaveData(file9, name=intensity9,PixPerMicron=4.5429)
Ng_plastic_159e9 = ShockwaveData(file10, name=intensity10,PixPerMicron=4.5429)

t_s = Ng_Ta_169e9.getTimes().to_numpy() * 1e-9
log_t = np.log(t_s)

dist = Ng_Ta_169e9.getChannelPositions(2).to_numpy() * 1e-6
log_dist = np.log(dist) #NEED to make sure you remove weird distances, before the true zero time

#print(Ta_190e15.getTimes().to_numpy())
#print(t_s)
#print(Ta_190e15.getChannelPositions(2).to_numpy())
#print(dist)
#print(log_dist)
#print(log_t)
#print (Ta_551e15.getChannelPositions(1).to_numpy())
#print (Ta_551e15.getChannelPositions(2).to_numpy())

#exit()

sampleList = [Ta_551e15,Ta_283e15, Ta_190e15, Plastic_692e15, Plastic_653e15]
#sampleList = [Plastic_256e15, Plastic_653e15, Plastic_692e15]
#sampleList = [Ta_190e15,Ta_283e15,Ta_551e15,Ng_Ta_102e9,Ng_Ta_169e9]
#sampleList = [Plastic_653e15, Plastic_692e15,Ng_plastic_102e9,Ng_plastic_159e9]
#sampleList = [Ng_Ta_169e9, Ng_Ta_102e9, Ng_plastic_159e9, Ng_plastic_102e9]
#sampleList = [Ng_plastic_102e9,Ng_plastic_159e9]
# sampleList = [Ta234]
# %%
# Generation of plots


#def func(log_t, C, m):
#    return (C + m*log_t)
#p0 = np.array([1.0e-6,3.]) #initial guesses for the coefficients

'curve_fit(f, xdata, ydata, p0=None)'

#popt,pcov = curve_fit(func,log_t,log_dist,p0,maxfev = 10000)
#print('coeff are:',popt)

#m = popt[1]
#Beta = 2/m - 2
#print(f"beta = {Beta}")

#'sklearn.metrics.r2_score(y_true, y_pred)'
#r2 = r2_score(log_dist,func(log_t,*popt))
#print('r^2 = ', r2)

#plt.plot(log_t,log_dist,label='data',c='r',marker="o",linestyle="")
#plt.plot(log_t, func(log_t, *popt),label='fit',c='k')
#plt.xlabel('log(time) [log(s)]')
#plt.ylabel('log(distance) [log(m)]')
#plt.legend(loc='best')
#plt.show()

#exit()

colors = ["b", "g", "r", "c", "m", "y", "k"]


def plotChannelPositionandFit(data: ShockwaveData, channel, color=None):
    times = np.linspace(data.getTimes().min(), data.getTimes().max(), 100)
    plt.plot(times, data.positionPolynomial(1)(times), color=color)
    plt.plot(
        data.getTimes().to_numpy(),
        data.getChannelPositions(1).to_numpy(),
        linestyle="",
        marker="o",
        label=data.getName(),
        color=color,
    )
    plt.legend()
    plt.xlabel("Time (ns)")
    plt.ylabel("Position (µm)")
    plt.title("Propagation of Shockwave")


# Need to output numpy lists
# Justin was here 4/10/25
def plotVelocityvsSingleImage(data: ShockwaveData, channel: int, color=None):
    times = np.linspace(data.getTimes().min(), data.getTimes().max(), 100)
    plt.plot(
        times,
        data.velocityPolynomial(channel)(times),
        color=color,
        label=f"{data.getName()} multi-image velocity",
    )

    plt.plot(
        data.getTimes().to_numpy(),
        data.singleImageVelocities(last_channel=4).to_numpy(),
        marker="o",
        linestyle="",
        label=f"{data.getName()} single-image velocity",
        color=color,
    )
    plt.legend()
    plt.xlabel("Time (ns)")
    plt.ylabel("Velocity (µm/s)")
    plt.title("Shockwave Speed Single vs. Multi image")


def plotAllChannelPositions(data: ShockwaveData):
    for i in range(1, data.getNumberOfChannels() + 1):
        plt.plot(
            data.getTimes().to_numpy(),
            data.getChannelPositions(i).to_numpy(),
            marker="o",
            linestyle="",
            label=f"Channel {i}",
        )
    plt.xlabel("Time (ns)")
    plt.ylabel("Position (µm)")
    plt.legend()
    plt.title(f"{data.getName()} Channel vs. Time Positions")


for shockwave in sampleList:
    plotAllChannelPositions(shockwave)
    plt.show()


for i, shockwavedata in enumerate(sampleList):
    plotChannelPositionandFit(shockwavedata, 1, color=colors[i])
plt.show()

for i, shockwavedata in enumerate(sampleList):
    plotVelocityvsSingleImage(shockwavedata, 1, color=colors[i])
plt.show()



# %% Fit and plot log-log distance vs. time for all samples

def log_log_fit_and_plot(samples, channel=2):
    plt.figure(figsize=(10, 6))

    for i, sample in enumerate(samples):
        times = sample.getTimes(channel).to_numpy() * 1e-9  # in seconds
        dists = sample.getChannelPositions(channel).to_numpy() * 1e-6  # in meters

        # Remove non-positive values
        valid = (times > 0) & (dists > 0)
        times = times[valid]
        dists = dists[valid]

        if len(times) < 3:
            print(f"Skipping {sample.getName()} due to insufficient valid data points.")
            continue

        log_t = np.log(times)
        log_d = np.log(dists)

        # Fit log-log data
        def func(log_t, C, m):
            return C + m * log_t

        p0 = [1.0e-6, 3]
        try:
            popt, pcov = curve_fit(func, log_t, log_d, p0=p0, maxfev=10000)
            m = popt[1]
            beta = 2 / m - 2
            r2 = r2_score(log_d, func(log_t, *popt))

            print(f"{sample.getName()} — C = {popt[0]:.3f}, m = {m:.3f}, β = {beta:.3f}, R² = {r2:.4f}")

            # Plot
            color = colors[i % len(colors)]
            plt.plot(log_t, log_d, marker='o', linestyle='', label=f"{sample.getName()} data", color=color)
            plt.plot(log_t, func(log_t, *popt), linestyle='-', label=f"{sample.getName()} fit", color=color)
        except RuntimeError:
            print(f"Fit failed for {sample.getName()}.")

    plt.xlabel("log(time) [log(s)]")
    plt.ylabel("log(distance) [log(m)]")
    plt.title("log-log Fit of Shockwave Propagation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


log_log_fit_and_plot(sampleList, channel=2)

# %%
# can calculate acceleration at some point too
