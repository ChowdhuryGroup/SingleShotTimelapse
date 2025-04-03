# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filepath = "data/Ta 455mw shockwave.tsv"


class ShockwaveData:
    def __init__(self, filepath, PixPerMicron=4.04):
        self.df = pd.read_csv(filepath, sep="\t")
        self.time = self.df["time"]
        self.positions = self.df.iloc[:, 1:] / PixPerMicron
        self.probeTimes = [0, 1, 2, 3]  # ns times of when probe channels arrive

    def imageToImageVelocity(self, channel):
        # calculates difference of position on "channel" from image 1 to image 3 divided by the time between image 1 and 3
        time_diff = self.time.shift(-1) - self.time.shift(1)
        position_diff = self.positions[f"channel{channel}X coord"].shift(
            -1
        ) - self.positions[f"channel{channel}X coord"].shift(1)
        velocity = position_diff / time_diff
        return velocity.iloc[1:-1]

    def positionPolynomial(self, channel, degree=6):
        coefficients = np.polyfit(
            self.getTimes(), self.getChannelPositions(channel), degree
        )
        return np.poly1d(coefficients)

    def velocityPolynomial(self, channel):
        return np.polyder(self.positionPolynomial(channel))

    def singleImageVelocities(self, first_channel=1, last_channel=4):
        velocity_in_single_image = (
            df[f"channel{first_channel}X coord"] - df[f"channel{last_channel}X coord"]
        ) / (self.probeTimes[last_channel - 1] - self.probeTimes[first_channel - 1])
        return velocity_in_single_image

    def getChannelPositions(self, channel):
        return self.positions[f"channel{channel}X coord"]

    def getTimes(self):
        return self.time


filepath = "data/Ta 455mw shockwave.tsv"
Ta455 = ShockwaveData(filepath)
times = np.linspace(Ta455.getTimes().min(), Ta455.getTimes().max(), 100)
plt.plot(times, Ta455.positionPolynomial(1)(times))
plt.plot(Ta455.getTimes(), Ta455.getChannelPositions(1), linestyle="", marker="o")
plt.show()

plt.plot(times, Ta455.velocityPolynomial(1))
plt.plot(Ta455.getTimes(), Ta455.imageToImageVelocity(1))

# %%

df = pd.read_csv(filepath, sep="\t")
single_image_velocity = (df["channel1X coord"] - df["channel4X coord"]) / 4


for i in range(1, 5):
    plt.scatter(df["time"], df[f"channel{i}X coord"] / 4.04, label=f"channel {i}")
plt.xlabel("time (on channel 1) (ns)")
plt.ylabel("distance (µm)")
plt.title("Shockwave front Glass")
plt.legend()
plt.show()

# Calculate the average velocity for channel 1
time_diff = df["time"].shift(-1) - df["time"].shift(
    1
)  # Time difference between t[i+1] and t[i-1]
velocity_channel2 = (
    df["channel2X coord"].shift(-1) - df["channel2X coord"].shift(1)
) / time_diff

velocity_in_single_image = (df["channel1X coord"] - df["channel4X coord"]) / 4

# Plot the average velocity for channel 1
plt.scatter(
    df["time"].iloc[1:-1],
    velocity_channel2.iloc[1:-1] / 4.04,
    label="Channel 1 Velocity",
)
plt.scatter(
    df["time"],
    velocity_in_single_image / 4.04,
    label="Channel1-Channel4 velocity on single image",
)
plt.xlabel("time (on channel 1) (ns)")
plt.ylabel("Velocity (µm/ns)")
plt.title("Image to Image Velocity vs Single Image Velocity Glass")
plt.legend()
plt.show()

# %%
# Fit a polynomial of degree 2 (you can adjust the degree as needed)
degree = 6
coefficients = np.polyfit(df["time"], df["channel2X coord"], degree)
position_polynomial = np.poly1d(coefficients)


# Plot the original positions with fitted polynomial
plt.scatter(df["time"], df["channel2X coord"] / 4.04, label="Original Data")
plt.plot(
    df["time"],
    position_polynomial(df["time"]) / 4.04,
    label="Fitted Polynomial",
    color="red",
)
plt.xlabel("Time (ns)")
plt.ylabel("Position (µm)")
plt.title("Shockwave Position and Fitted Polynomial")
plt.legend()
plt.show()

# %%
# now we need to take the derivative
velocity_polynomial = np.polyder(position_polynomial)
velocities = velocity_polynomial(df["time"].iloc[1:-1])

# Plot the velocity
plt.plot(
    df["time"].iloc[1:-1],
    velocities / 4.04,
    marker="o",
    linestyle="-",
    label="Image to Image Velocity",
)
plt.plot(
    df["time"].iloc[1:-1],
    velocity_in_single_image.iloc[1:-1] / 4.04,
    marker="o",
    linestyle="",
    label="Single Shot Velocity",
)
plt.xlabel("Time (ns)")
plt.ylabel("Velocity (µm/ns)")
plt.title("458mW in Glass Single Shot Velocity vs. Image to Image velocity")
plt.legend()
plt.show()


# It would be cool to plot instantaneous velocity in one image vs avg velocity between images x[i+1]-x[i]-x[i-1]/3 average over 3 points

# %%
