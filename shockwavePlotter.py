import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filepath = "data/glass Air 458mw.tsv"


df = pd.read_csv(filepath, sep="\t")
single_image_velocity = (df["channel1X coord"] - df["channel4X coord"]) / 4


for i in range(1, 5):
    plt.scatter(df["time"], df[f"channel{i}X coord"], label=f"channel {i}")
plt.xlabel("time (ns)")
plt.ylabel("Pixel Location")
plt.legend()
plt.show()

# It would be cool to plot instantaneous velocity in one image vs avg velocity between images x[i+1]-x[i]-x[i-1]/3 average over 3 points
