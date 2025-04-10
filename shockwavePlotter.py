# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np


# %%
# Class to handle loading and fitting of positions
class ShockwaveData:
    def __init__(self, filepath, PixPerMicron=4.04, name=None):
        self.df = pd.read_csv(filepath, sep="\t")
        self.time = self.df["time"]
        self.positions = self.df.iloc[:, 1:] / PixPerMicron
        self.numberChannels = self.positions.shape[1]
        self.probeTimes = [0, 1, 2, 3]  # ns times of when probe channels arrive
        self.name = name if name is not None else filepath

    def imageToImageVelocity(self, channel):
        # calculates difference of position on "channel" from image 1 to image 3 divided by the time between image 1 and 3
        time_diff = self.time.shift(-1) - self.time.shift(1)
        position_diff = self.positions[f"channel{channel}X coord"].shift(
            -1
        ) - self.positions[f"channel{channel}X coord"].shift(1)
        velocity = position_diff / time_diff
        return velocity.iloc[1:-1]

    def positionPolynomial(self, channel, degree=3):
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

    def getTimes(self):
        return self.time

    def getName(self):
        return self.name

    def getNumberOfChannels(self):
        return self.numberChannels


# %%
# Loading in Data Files
file1, intensity1 = "data/Ta 520mw shockwave.tsv", "Ta 6.66e15W/cm^2"
file2, intensity2 = "data/Ta 455mw shockwave.tsv", "Ta 5.51e15W/cm^2"
file3, intensity3 = "data/Ta 234mw shockwave.tsv", "Ta 2.83e15W/cm^2"
file4, intensity4 = "data/Ta 157mw shockwave.tsv", "Ta 1.90e15W/cm^2"
file5, intensity5 = "data/plasticTa 505mw shockwave.tsv", "Plastic Ta "

Ta520 = ShockwaveData(file1, name=intensity1)
Ta455 = ShockwaveData(file2, name=intensity2)
Ta234 = ShockwaveData(file3, name=intensity3)
Ta157 = ShockwaveData(file4, name=intensity4)
PlasticTa = ShockwaveData(file5, name=intensity5)

sampleList = [Ta520, Ta455, Ta234, Ta157]
# sampleList = [Ta455, Ta234, Ta157]
sampleList = [Ta455, Ta234, Ta157]
# sampleList = [Ta234]
# %%
# Generation of plots

colors = ["b", "g", "r", "c", "m", "y", "k"]


def plotChannelPositionandFit(data: ShockwaveData, channel, color=None):
    times = np.linspace(data.getTimes().min(), data.getTimes().max(), 100)
    plt.plot(times, data.positionPolynomial(1)(times), color=color)
    plt.plot(
        data.getTimes(),
        data.getChannelPositions(1),
        linestyle="",
        marker="o",
        label=data.getName(),
        color=color,
    )
    plt.legend()
    plt.xlabel("Time (ns)")
    plt.ylabel("Position (µm)")
    plt.title("Propogation of Shockwave")


def plotVelocityvsSingleImage(data: ShockwaveData, channel: int, color=None):
    times = np.linspace(data.getTimes().min(), data.getTimes().max(), 100)
    plt.plot(
        times,
        data.velocityPolynomial(channel)(times),
        color=color,
        label=f"{data.getName()} multi-image velocity",
    )

    plt.plot(
        data.getTimes(),
        data.singleImageVelocities(last_channel=4),
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
            data.getTimes(),
            data.getChannelPositions(i),
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

# %%
# can calculate acceleration at some point too
