import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import math
from matplotlib.ticker import LogLocator
plt.rcParams['font.size'] = 20 # Sets the font size to 16

ro = 1.293 #kg/m^3
MarkerList = ['s', 'o', 'D', 'x', 'p']
LineList = ['-', ':', '--', '-.', '-']
std_dev = 2.71/4.04 #now in um, error from manually selecting edge of shockwave

plt.rcParams.update(
    {
        # Text and Font
        "font.family": "serif",  # Professional serif font
        "font.size": 12,  # Base font size
        "axes.labelsize": 14,  # X and Y label size
        "axes.titlesize": 14,  # Title size
        "legend.fontsize": 10,  # Legend text
        "xtick.labelsize": 10,  # Axis tick labels
        "ytick.labelsize": 10,
        # Line and Marker Styles
        "lines.linewidth": 2.0,  # Thicker lines for visibility
        "lines.markersize": 6,  # Larger markers
        "axes.linewidth": 1.2,  # Thicker axis frame
        # Layout and Exporting
        "figure.figsize": (6, 4),  # Standard aspect ratio
        # "figure.autolayout": True,    # Same as plt.tight_layout()
        "savefig.dpi": 300,  # High-resolution export
        "savefig.bbox": "tight",  # No extra white space on export
        # Colors and Grid
        "axes.grid": False,  # Subtle grid helps data reading
        "grid.alpha": 0.3,  # Make grid faint
        "grid.linestyle": "--",
        "axes.prop_cycle": plt.cycler(
            color=[
        "#004488",  # Dark Blue
        "#DDAA33",  # Gold/Yellow
        "#BB5566",  # Rose/Red
        "#000000",  # Black
        "#6699CC",  # Light Blue
        "#EE99AA",  # Pink
        "#994499"   # Purple
    ]
        ),
    }
)


# %%
# Class to handle loading and fitting of positions
class ShockwaveData:
    def __init__(self, filepath, PixPerMicron=4.04, name=None):
        self.df = pd.read_csv(filepath, sep="\t")
        self.time = self.df["time"]
        self.positions = self.df.iloc[:, 1:] / PixPerMicron
        self.numberChannels = self.positions.shape[1]
        self.probeTimes = [0, 1, 2.5, 3.5]  # ns times of when probe channels arrive
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
    

    def variancePolynomial(self, channel):
        def line(x, m, b):
            return m * x + b
        popt, pcov = curve_fit(line, self.getTimes(), self.getChannelPositions(channel=channel))
        # Note: Despite the name, this returns standard error of the slope parameter
        # pcov[0,0] is variance of slope parameter m, sqrt gives standard error
        return np.sqrt(pcov[0,0])

    def velocityPolynomial(self, channel):
        return np.polyder(self.positionPolynomial(channel))

    def singleImageVelocities(self, first_channel=1, last_channel=4):
        velocity_in_single_image = (
            self.getChannelPositions(first_channel)
            - self.getChannelPositions(last_channel)
        ) / (self.probeTimes[last_channel - 1] - self.probeTimes[first_channel - 1])

        #um_error = abs(velocity_in_single_image)#*np.sqrt(std_dev**2/(self.getChannelPositions(first_channel) - self.getChannelPositions(last_channel))**2)
        error = np.sqrt(std_dev**2/(self.getChannelPositions(first_channel) - self.getChannelPositions(last_channel))**2)
        um_error = abs(velocity_in_single_image) * error
        return velocity_in_single_image, um_error

    def getChannelPositions(self, channel):
        return self.positions[f"channel{channel}X coord"]

    def getTimes(self,channel=1):
        return self.time+self.probeTimes[channel]

    def getName(self):
        return self.name

    def getNumberOfChannels(self):
        return self.numberChannels
    
    def localMultiImageVelocity(self, channel):
        times = self.getTimes(channel).to_numpy()
        positions = self.getChannelPositions(channel).to_numpy()

        # Clean data
        valid = (~np.isnan(times)) & (~np.isnan(positions))
        times = times[valid]
        positions = positions[valid]

        # Calculate finite differences
        dt = np.diff(times)
        dp = np.diff(positions)
        v_local = dp / dt  # velocity at midpoints# Midpoints of time for plotting
        t_mid = (times[:-1] + times[1:]) / 2
        
        # Calculate error from position measurement uncertainty
        # std_dev (defined at module level) is the position measurement error in µm
        # Error propagation: δv = sqrt((δp1/dt)^2 + (δp2/dt)^2) = sqrt(2) * std_dev / dt
        v_error = np.sqrt(2) * std_dev / dt

        return t_mid, v_local, v_error


# %%
# Loading in Data Files
file1, intensity1 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ta_190e15.tsv", r"fs, Ta 1.90e15W/cm$^2$"
file2, intensity2 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ta_283e15.tsv", r"fs, Ta 2.83e15W/cm$^2$"
file3, intensity3 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ta_551e15.tsv", r"fs, Ta 5.51e15W/cm$^2$"
file4, intensity4 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Plastic_256e15.tsv", r"fs, Plastic 2.56e15W/cm$^2$"
file5, intensity5 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Plastic_653e15.tsv", r"fs, Plastic 6.53e15W/cm$^2$"
file6, intensity6 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Plastic_692e15.tsv", r"fs, Plastic 6.92e15W/cm$^2$"
file7, intensity7 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_Ta_102e9.tsv", r"ns, Ta 1.40e9W/cm$^2$"
file8, intensity8 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_Ta_169e9.tsv", r"ns, Ta 2.33e9W/cm$^2$"
file9, intensity9 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_plastic_102e9.tsv", r"ns, Plastic 1.40e9W/cm$^2$"
file10, intensity10 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Ng_plastic_159e9.tsv", r"ns, Plastic 2.19e9W/cm$^2$"
file11, intensity11 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Transv_Glass_554e15_air.tsv", r"Glass 5.54e15W/cm$^2$ IN AIR"
file12, intensity12 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Transv_Glass_554e15_glass.tsv", r"Glass 5.54e15W/cm$^2$ IN BULK"
file13, intensity13 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Transv_Glass_190e15_air.tsv", r"Glass 1.90e15W/cm$^2$ IN AIR"
file14, intensity14 = r"C:\Users\tward\OneDrive\Desktop\Wszystko\Praca\Spectral Energies\03012025 SE expt\Transv_Glass_190e15_glass.tsv", r"Glass 1.90e15W/cm$^2$ IN BULK"


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
Glass_554e15_air = ShockwaveData(file11, name=intensity11) #in transverse orientation
Glass_554e15_glass = ShockwaveData(file12, name=intensity12) #in transverse orientation
Glass_190e15_air = ShockwaveData(file13, name=intensity13) #in transverse orientation
Glass_190e15_glass = ShockwaveData(file14, name=intensity14) #in transverse orientation


t_s = Ng_Ta_169e9.getTimes().to_numpy() * 1e-9
log_t = np.log(t_s)

dist = Ng_Ta_169e9.getChannelPositions(2).to_numpy() * 1e-6
log_dist = np.log(dist) #NEED to make sure you remove weird distances, before the true zero time


#sampleList = [Ta_551e15,Ta_283e15, Ta_190e15, Plastic_692e15, Plastic_653e15]
sampleList = [Ta_283e15]
#sampleList = [Ta_190e15,Ta_283e15,Ta_551e15,Ng_Ta_102e9,Ng_Ta_169e9]
#sampleList = [Plastic_653e15, Plastic_692e15,Ng_plastic_102e9,Ng_plastic_159e9]
#sampleList = [Ng_Ta_169e9, Ng_Ta_102e9, Ng_plastic_159e9, Ng_plastic_102e9]
#sampleList = [Ng_plastic_102e9,Ng_plastic_159e9]
#sampleList = [Ta_551e15, Plastic_692e15, Ng_Ta_169e9, Ng_plastic_159e9]
#sampleList = [Glass_554e15_air, Glass_190e15_air, Glass_554e15_glass, Glass_190e15_glass]
# sampleList = [Ta234]

# %%
# Generation of plots


colors = ["b", "g", "r", "c", "m", "y", "k"]
marker_dict = {'data1':'o', 'data2':'D'}


def plotChannelPositionandFit(data: ShockwaveData, channel, linestyle='', marker='o', color=None):
    #times = np.linspace(data.getTimes().min(), data.getTimes().max(), 100)
    times = data.getTimes(channel).to_numpy()
    positions = data.getChannelPositions(channel).to_numpy()

    # Remove NaNs or invalid points if needed
    valid = (~np.isnan(times)) & (~np.isnan(positions))
    times = times[valid]
    positions = positions[valid]

    # Polynomial fit (default is degree=1 for linear)
    degree = 1
    coefficients = np.polyfit(times, positions, degree)
    poly = np.poly1d(coefficients)
    fit_positions = poly(times)

    #Compute R^2
    r2 = r2_score(positions, fit_positions)
    
    # Compute standard error of the slope from the fit
    slope_std = data.variancePolynomial(channel)  # Returns std error despite function name

    # Output
    slope = coefficients[0]
    intercept = coefficients[1]
    print(f"{data.getName()} (Channel {channel}): Slope = {slope:.4f} ± {slope_std:.4f} µm/ns, Intercept = {intercept:.4f} µm, R² = {r2:.4f}")

    times_fit = np.linspace(times.min(), times.max(), 100)
    plt.plot(times_fit, poly(times_fit), linestyle=linestyle, color=color)
    plt.plot(times, positions, linestyle='', marker=marker, label=data.getName(), color=color)
    plt.legend()
    plt.legend(framealpha=0)
    plt.xlabel("Time (ns)")
    plt.xlim(-5,30)
    plt.ylabel("Position (µm)")
    #plt.title("Propagation of Shockwave")

    #plt.plot(times, data.positionPolynomial(1)(times), color=color)
    #plt.plot(
    #    data.getTimes().to_numpy(),
    #    data.getChannelPositions(1).to_numpy(),
    #    linestyle="",
    #    marker="o",
    #    label=data.getName(),
    #    color=color,
    #)
    #plt.legend()
    #plt.xlabel("Time (ns)")
    #plt.ylabel("Position (µm)")
    #plt.title("Propagation of Shockwave")


#trying to get instantaneous velocity in multi channel
def plotCombinedVelocities(data: ShockwaveData, channel: int, color=None):
    # --- Multi-image instantaneous velocity (finite difference) ---
    t_mid, v_local, v_error = data.localMultiImageVelocity(channel)
    plt.errorbar(t_mid, v_local, yerr=v_error, linestyle='', marker='o', color=color, label=f"{data.getName()}" + r" $v_{\text{inst}} \pm \sigma$")

    # --- Single-image velocity ---
    t_single = data.getTimes().to_numpy()
    v_single, v_err = data.singleImageVelocities(last_channel=4)
    v_single = v_single.to_numpy()
    v_err = v_err.to_numpy()
    plt.errorbar(t_single, v_single, yerr=v_err, linestyle='', marker='x', color='r', label=f"{data.getName()}" + r" $v_{\text{s}} \pm \sigma$")

    # --- Polynomial velocity fit ---
    t_poly = np.linspace(t_single.min(), t_single.max(), 100)
    v_poly = data.velocityPolynomial(channel)(t_poly)
    err_poly = data.variancePolynomial(channel)  # Returns std error despite function name
    #plt.plot(t_poly, v_poly, linestyle='--', color='k', label=f"{data.getName()}" + r" $v_{\text{avg}}$")

    plt.fill_between(t_poly, v_poly - err_poly, v_poly + err_poly, color='k', label=f"{data.getName()}" + r" $v_{\text{avg}} \pm \sigma$", alpha=0.5)

    # --- Labels and legend ---
    plt.xlabel("Time (ns)")
    plt.ylabel("Velocity (km/s)")
    #plt.ylim((0,20))
    #plt.title("Shockwave Velocity calculated with various methods")
    plt.legend()
    plt.legend(framealpha=0)

for i, shockwavedata in enumerate(sampleList):
    plotCombinedVelocities(shockwavedata, channel=1, color=colors[i])
plt.show()


# Need to output numpy lists
# Justin was here 4/10/25
def plotVelocityvsSingleImage(data: ShockwaveData, channel: int, color=None):
    times = np.linspace(data.getTimes().min(), data.getTimes().max(), 100)
    v_poly = data.velocityPolynomial(channel)(times)
    err_poly = data.variancePolynomial(channel)  # Returns std error despite function name; scalar constant error band
    
    plt.plot(
        times,
        v_poly,
        color=color,
        label=f"{data.getName()} multi-image velocity ± σ",
    )
    plt.fill_between(times, v_poly - err_poly, v_poly + err_poly, color=color, alpha=0.3)

    t_single = data.getTimes().to_numpy()
    v_single, v_err = data.singleImageVelocities(last_channel=4)
    v_single = v_single.to_numpy()
    v_err = v_err.to_numpy()
    
    plt.errorbar(
        t_single,
        v_single,
        yerr=v_err,
        marker="o",
        linestyle="",
        label=f"{data.getName()} single-image velocity ± σ",
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
    plotChannelPositionandFit(shockwavedata, 1, linestyle=LineList[i], marker=MarkerList[i], color=colors[i])
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

        log_t = np.log(times)
        log_d = np.log(dists)

        # Fit log-log data
        def func(log_t, C, m):
            return C + m * log_t

        p0 = [1.0e-6, 3]
        try:
            loglog = True
            popt, pcov = curve_fit(func, log_t, log_d, p0=p0, maxfev=10000)
            m = popt[1]
            beta = 2 / m - 2
            r2 = r2_score(log_d, func(log_t, *popt))
            
            # Calculate error in m and propagate to beta
            # Error in m from covariance matrix
            m_std = np.sqrt(pcov[1, 1])
            # Error propagation: dβ/dm = -2/m², so δβ = (2/m²) * δm
            beta_std = (2 / m**2) * m_std

            print(f"{sample.getName()} — C = {popt[0]:.3f}, m = {m:.3f} ± {m_std:.3f}, β = {beta:.3f} ± {beta_std:.3f}, R² = {r2:.4f}")

            # Plot
            color = colors[i % len(colors)]
            if loglog:
                plt.plot(np.exp(log_t), np.exp(log_d), marker=MarkerList[i], linestyle='', color=color)
                plt.plot(np.exp(log_t), np.exp(func(log_t, *popt)), linestyle=LineList[i], label=f"{sample.getName()}, β = {beta:.2f} ± {beta_std:.2f}", color=color)
            else:
                plt.plot(log_t, log_d, marker='o', linestyle='', label=f"{sample.getName()} data", color=color)
                plt.plot(log_t, func(log_t, *popt), linestyle='-', label=f"{sample.getName()} fit, β = {beta:.2f} ± {beta_std:.2f}", color=color)
        except RuntimeError:
            print(f"Fit failed for {sample.getName()}.")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    if loglog:
        plt.loglog()
        ymin, ymax = plt.ylim()
        plt.ylim((1e-4, ymax))
    #plt.title("log-log Fit of Shockwave Propagation")
    plt.legend(framealpha=0)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()


log_log_fit_and_plot(sampleList, channel=2)

# %%
# can calculate acceleration at some point too
