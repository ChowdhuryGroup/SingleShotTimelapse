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
velocity_channel1 = (
    df["channel1X coord"].shift(-1) - df["channel1X coord"].shift(1)
) / time_diff

velocity_in_single_image = (df["channel1X coord"] - df["channel4X coord"]) / 4

# Plot the average velocity for channel 1
plt.scatter(
    df["time"].iloc[1:-1],
    velocity_channel1.iloc[1:-1] / 4.04,
    label="Channel 1 Velocity",
)
plt.scatter(
    df["time"],
    velocity_in_single_image,
    label="Channel1-Channel4 velocity on single image",
)
plt.xlabel("time (on channel 1) (ns)")
plt.ylabel("Velocity (µm/ns)")
plt.title("Image to Image Velocity vs Single Image Velocity Glass")
plt.legend()
plt.show()

# It would be cool to plot instantaneous velocity in one image vs avg velocity between images x[i+1]-x[i]-x[i-1]/3 average over 3 points
