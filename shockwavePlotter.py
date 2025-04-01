# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filepath = "data/glass glass 458mw.tsv"


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
plt.scatter(df["time"], df["channel2X coord"], label="Original Data")
plt.plot(
    df["time"], position_polynomial(df["time"]), label="Fitted Polynomial", color="red"
)
plt.xlabel("Time (ns)")
plt.ylabel("velocity µm/s")
plt.title("Shockwave velocity and Fitted Polynomial")
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
    linestyle="-",
    label="Single Shot Velocity",
)
plt.xlabel("Time (ns)")
plt.ylabel("Velocity (µm/ns)")
plt.title("Single Shot Velocity vs. Image to Image velocity")
plt.legend()
plt.show()


# It would be cool to plot instantaneous velocity in one image vs avg velocity between images x[i+1]-x[i]-x[i-1]/3 average over 3 points

# %%
