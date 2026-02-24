"""

Author: Conrad Kuz
2025-04-13

"""

# Import the .NET class library
import clr

# Import python sys module
import sys

# Import os module
import os
import time
from ChowdhuryLabDevices.oscilloscope import Oscilloscope
from ChowdhuryLabDevices.DG645_Delay_Generator import DG645

# Import System.IO for saving and opening files
from System.IO import *

# Import c compatible List and String
from System import String
from System.Collections.Generic import List

# Add needed dll references
sys.path.append(os.environ["LIGHTFIELD_ROOT"])
sys.path.append(os.environ["LIGHTFIELD_ROOT"] + "\\AddInViews")
clr.AddReference("PrincetonInstruments.LightFieldViewV5")
clr.AddReference("PrincetonInstruments.LightField.AutomationV5")
clr.AddReference("PrincetonInstruments.LightFieldAddInSupportServices")

# PI imports
from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import ExperimentSettings
from PrincetonInstruments.LightField.AddIns import DeviceType


mainDirectory = "C:\\Users\\twardowski.6a\\Documents\\SEProbe\\2025-04-16"
trigCom = "COM4"

probeDelay = 1e-3  # In seconds
pumpDelay = 0.00116474  # In seconds


# Check if given directory exists
if not os.path.exists(mainDirectory):
    os.makedirs(mainDirectory)


def device_found():
    # Find connected device
    for device in experiment.ExperimentDevices:
        if device.Type == DeviceType.Camera:
            return True

    # If connected device is not a camera inform the user
    print("Camera not found. Please add a camera and try again.")
    return False


# Create the LightField Application (true for visible)
# The 2nd parameter forces LF to load with no experiment
auto = Automation(True, List[String]())

# Get experiment object
experiment = auto.LightFieldApplication.Experiment

# will need to change this to timing correct NG
experiment.Load("SE triggered right off diode")


def changeCameraDirectory(newDirectory: str):
    r"""This will change light field to a new directory, makes directory if given does not exist
    newDirectory : String, example: "c:\Users\twardowski.6a\Documents\SEprobe" """

    if not os.path.exists(newDirectory):
        os.makedirs(newDirectory)
    experiment.SetValue(
        ExperimentSettings.OnlineExportOutputOptionsCustomDirectory, newDirectory
    )


def readyCamera(saveDirectory: str):
    experiment.Stop()
    time.sleep(.1)
    changeCameraDirectory(saveDirectory)
    experiment.Acquire()


trigBox = DG645(trigCom)
scope = Oscilloscope()

timing_file_path = os.path.join(mainDirectory, "timings.txt")


def log_trial_time(filepath, trial_number, channelTime):
    """
    Appends a new trial entry to the timings.tsv file.
    Adds a header if the file doesn't exist yet.
    """
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a") as f:
        if not file_exists:
            f.write("trial_number\ttime(ns)\n")  # header
        f.write(f"{trial_number}\t{channelTime*1e9}\n")


while True:
    choice = input(
        "enter for data collection, b for before only, d for during only, c to change timing, q to quit: "
    )
    if choice == "q":
        # Exit program when user quits
        break
    if choice == "c":
        timeChange = input("Current Timing +/- x seconds, x=")
        pumpDelay += float(timeChange)
        print(f"New pump delay: {pumpDelay}")

    # This section makes new trial folders
    folder_numbers = []
    for name in os.listdir(mainDirectory):
        path = os.path.join(mainDirectory, name)
        if os.path.isdir(path) and name.isdigit():
            folder_numbers.append(int(name))

    # Get the highest folder number
    highestFolder = 0
    if folder_numbers:
        highestFolder = max(folder_numbers)

    trialFolderPath = os.path.join(mainDirectory, f"{(highestFolder+1):03}")
    os.mkdir(trialFolderPath)

    # Set camera to acquire mode with trial file path
    #Is there a way to pause the camera when its running?
    # send probe in only
    if choice != "d":
        readyCamera(trialFolderPath)
        time.sleep(0.5)
        trigBox.disableChannel(2)
        trigBox.setOutputTimeandWidth(1, probeDelay)
        trigBox.trigger()
        print("Before Done")
        print("SE 5 second Timer")
        time.sleep(5)

    if choice != "b":
        scope.select_channels((1, 2))  # Channel 1 and 2 on scope
        scope.start_single_acquisition(380e-9)  # set width of acquisition
        readyCamera(trialFolderPath)
        time.sleep(1)
        trigBox.setOutputTimeandWidth(1, probeDelay)
        trigBox.setOutputTimeandWidth(2, pumpDelay)
        trigBox.trigger()
        scope.read_acquisition()
        log_trial_time(
            timing_file_path, highestFolder + 1, scope.get_channel_max_time(1)
        )
        print(f"Scope Channel 2 max: {scope.get_channel_max_time(1)}")
        print("During Done")
        print("SE 5 second Timer")
        time.sleep(5)
    
    
# potentially show a composite image?
#I should just make pump probe viewer a class and import it?
#Maybe i should just use the already compiled resource?


