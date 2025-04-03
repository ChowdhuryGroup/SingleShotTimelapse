import os
import pandas as pd
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog


mpl.use("TkAgg")  # Need this to work when selecting points in ginput (on mac at least?)

# User Inputs
directory = "data/TA/234mw"
darkFieldPath = "data/glass/bkgWithFlash.tif"
darkFieldPath = "data/TA/bkgCameraBlocked.tif"
zero_time = 56
sample_is_glass = False


tsv_file = os.path.join(directory, "timings.txt")

# load all images to corresponding times
df = pd.read_csv(tsv_file, sep="\t", header=0, names=["trial", "time"])

# filter out lost timing trials
df = df[pd.to_numeric(df["time"], errors="coerce").notnull()]

darkbkg = cv2.imread(darkFieldPath, cv2.IMREAD_ANYDEPTH).astype(np.float32)

before_images_by_time = {}
during_images_by_time = {}
normalized_images_by_time = {}
edge_positions = {}  # Pixel column that the sample edge is in

# Iterate through each trial folder
for trial in df["trial"]:
    trial_folder = os.path.join(directory, str(trial).zfill(2))
    image_files = [f for f in os.listdir(trial_folder) if f.endswith(".tif")]

    # Sort image files by date and time in the filename
    image_files.sort(
        key=lambda x: datetime.strptime(x.split("/")[-1], "%Y %B %d %H_%M_%S.tif")
    )
    # Load images using cv2
    before_image_path = os.path.join(trial_folder, image_files[0])
    during_image_path = os.path.join(trial_folder, image_files[1])

    before_image = cv2.imread(before_image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    during_image = cv2.imread(during_image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

    before_image = np.clip(before_image, 0.1, 65535)
    during_image = np.clip(during_image, 0.1, 65535)

    normalized_image = during_image / before_image

    # Get sample edge:
    # Sum the pixel values along the columns
    column_sum = np.sum(before_image, axis=0)

    # Define the window size
    window_size = 10

    # Calculate the gradient within the window
    gradients = np.array(
        [
            np.gradient(column_sum[i : i + window_size])
            for i in range(len(column_sum) - window_size + 1)
        ]
    )
    max_gradients = np.max(np.abs(gradients), axis=1)

    # Find the position of the biggest gradient
    edge_position = np.argmax(max_gradients)

    # Center the images based on the edge position

    original_image_width = before_image.shape[0]
    shift = original_image_width - edge_position

    def center_image(image, shift):
        if shift > 0:
            image = np.pad(image, ((0, 0), (shift, 0)), mode="constant")
        elif shift < 0:
            image = np.pad(image, ((0, 0), (0, -shift)), mode="constant")
        return image

    # Check if the sample is glass

    if sample_is_glass:
        # Center the edge position with the whole image on display
        center_position = before_image.shape[0] // 2
        shift = center_position - edge_position

        def align_image(image, shift):
            if shift > 0:
                image = np.pad(image, ((0, 0), (shift, 0)), mode="constant")[
                    :, : image.shape[0]
                ]
            elif shift < 0:
                image = np.pad(image, ((0, 0), (0, -shift)), mode="constant")[
                    :, -shift:
                ]
            return image

        before_image = align_image(before_image, shift)
        during_image = align_image(during_image, shift)
        normalized_image = align_image(normalized_image, shift)

        # Crop the images to ensure they have the same width
        crop_width = before_image.shape[0]
        before_image = before_image[:, :crop_width]
        during_image = during_image[:, :crop_width]
        normalized_image = normalized_image[:, :crop_width]
    else:
        # Move the sample to the edge
        original_image_width = before_image.shape[0]
        shift = original_image_width - 60 - edge_position

        def align_image(image, shift):
            if shift > 0:
                image = np.pad(image, ((0, 0), (shift, 0)), mode="constant")
            elif shift < 0:
                image = np.pad(image, ((0, 0), (0, -shift)), mode="constant")
            return image

        before_image = align_image(before_image, shift)
        during_image = align_image(during_image, shift)
        normalized_image = align_image(normalized_image, shift)

        # Crop and flip images to same size
        before_image = np.fliplr(before_image[:, :original_image_width])
        during_image = np.fliplr(during_image[:, :original_image_width])
        normalized_image = np.fliplr(normalized_image[:, :original_image_width])

    # Get the corresponding time for the trial
    time_of_image = zero_time - float(df.loc[df["trial"] == trial, "time"].iloc[0])

    # Store images in the dictionaries
    if time_of_image not in before_images_by_time:
        before_images_by_time[time_of_image] = [before_image]
        during_images_by_time[time_of_image] = [during_image]
        normalized_images_by_time[time_of_image] = [normalized_image]
        edge_positions[time_of_image] = [edge_position]
    else:
        before_images_by_time[time_of_image].append(before_image)
        during_images_by_time[time_of_image].append(during_image)
        normalized_images_by_time[time_of_image].append(normalized_image)
        edge_positions[time_of_image].append(edge_position)


# Sort the keys
sorted_keys = sorted(normalized_images_by_time.keys())


def showAnmiation():
    # Create figure
    fig, ax = plt.subplots()

    # Function to update the frame
    def update_frame(i):
        ax.clear()
        frame_min = normalized_images_by_time[sorted_keys[i]][0].min()
        frame_max = normalized_images_by_time[sorted_keys[i]][0].max()
        frame_min = 0
        frame_max = 4
        print(frame_min, frame_max)
        ax.imshow(
            normalized_images_by_time[sorted_keys[i]][0],
            cmap="gray",
            vmin=frame_min,
            vmax=frame_max,
        )
        ax.set_title(f"Time: {sorted_keys[i]} , min/max{frame_max}")
        return ax

    # Create animation
    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(sorted_keys), interval=500
    )

    # Display the animation
    plt.show()


# Find the edge of sample for each channel in the before image


def showImages(imageDictionary, draw_line=True):
    global index
    # Initialize the index
    index = 0

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Display the first image
    image_display = ax.imshow(imageDictionary[sorted_keys[index]][0], cmap="gray")
    ax.set_title(
        f"Time: {sorted_keys[index]}, Edge pixel: {edge_positions[sorted_keys[index]][0]}"
    )
    if draw_line:
        ax.axvline(x=edge_positions[sorted_keys[index]][0], color="r", linestyle="--")

    # Function to update the image
    def update_image(event):
        global index
        index = (index + 1) % len(sorted_keys)
        ax.clear()
        image_display = ax.imshow(imageDictionary[sorted_keys[index]][0], cmap="gray")
        image_display.set_data(imageDictionary[sorted_keys[index]][0])
        ax.set_title(
            f"Time: {sorted_keys[index]}, Edge pixel: {edge_positions[sorted_keys[index]][0]}"
        )
        if draw_line:
            ax.axvline(
                x=edge_positions[sorted_keys[index]][0], color="r", linestyle="--"
            )

        plt.draw()

    # Create a button
    ax_button = plt.axes([0.45, 0.05, 0.1, 0.075])
    button = Button(ax_button, "Next")

    # Connect the button to the update function
    button.on_clicked(update_image)

    # Display the plot
    plt.show()


# showImages(before_images_by_time, draw_line=False)
# showImages(during_images_by_time, draw_line=False)
# showImages(normalized_images_by_time, draw_line=False)


# Save normalized images
def saveImages(imageDictionary, save_path):
    for key, image in imageDictionary.items():
        file_path = f"{save_path}/image_{key}_ns.tiff"
        cv2.imwrite(file_path, image[0].astype(np.float32))
        print(f"Saved {file_path}")


save_path = directory + "Compiled Images"
# os.makedirs(save_path)
# saveImages(normalized_images_by_time, save_path)


# Crop all images to each channel
number_of_channels = 4
timing_of_channels = [0, 1, 2, 3]  # top channel is timing of first value here

split_channels = {}

for key in sorted_keys:
    images_at_time = 0  # handle multiple images taken at the same time
    for image in normalized_images_by_time[key]:
        channels_in_image = {}
        channel_height = image.shape[1] // 4
        for channel in range(number_of_channels):
            channels_in_image[channel] = image[
                channel * channel_height : (channel + 1) * channel_height, :
            ]
        if key not in split_channels:
            split_channels[key] = [channels_in_image]
        else:
            split_channels[key].append(channels_in_image)


def showChannelsAndComposite(split_channels, number_of_channels, micron_per_pixel=1):
    sorted_keys = list(split_channels.keys())
    index = 0

    fig, axes = plt.subplots(1, number_of_channels + 1, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    def update_display():
        key = sorted_keys[index]
        for j in range(number_of_channels):
            axes[j].imshow(split_channels[key][0][j], cmap="gray")
            axes[j].set_title(f"Channel {j+1} at Time {round(key-j,2)}ns")
            axes[j].set_xticks(
                np.arange(0, split_channels[key][0][j].shape[1], step=220)
            )
            axes[j].set_xticklabels(
                [
                    round(x * micron_per_pixel, 2)
                    for x in np.arange(0, split_channels[key][0][j].shape[1], step=220)
                ]
            )
            axes[j].set_yticks([])  # Hide y-axis
            axes[j].set_xlabel("Distance (µm)")

        combined_image = np.vstack(
            [split_channels[key][0][j] for j in range(number_of_channels)]
        )
        axes[number_of_channels].imshow(combined_image, cmap="gray")
        axes[number_of_channels].set_title(
            f"Combined Image at Top Channel Time {round(key,2)}ns"
        )
        axes[number_of_channels].set_xticks(
            np.arange(0, combined_image.shape[1], step=220)
        )
        axes[number_of_channels].set_xticklabels(
            [
                round(x * micron_per_pixel, 2)
                for x in np.arange(0, combined_image.shape[1], step=220)
            ]
        )
        axes[number_of_channels].set_xlabel("Distance (µm)")
        axes[number_of_channels].set_yticks([])  # Hide y-axis

        plt.draw()

    def next_image(event):
        nonlocal index
        index = (index + 1) % len(sorted_keys)
        update_display()

    def prev_image(event):
        nonlocal index
        index = (index - 1) % len(sorted_keys)
        update_display()

    axprev = plt.axes([0.4, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.55, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")
    bprev = Button(axprev, "Previous")
    bnext.on_clicked(next_image)
    bprev.on_clicked(prev_image)

    update_display()
    plt.show()


# Example usage
# showChannelsAndComposite(split_channels, number_of_channels, micron_per_pixel=1 / 4.04)
# showChannelsAndComposite(split_channels, number_of_channels)


def showChannelsAndMarkFeatures(split_channels, number_of_channels):
    sorted_keys = list(split_channels.keys())
    index = 0
    markers = {}

    fig, axes = plt.subplots(1, number_of_channels + 1, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    def update_display():
        key = sorted_keys[index]
        for j in range(number_of_channels):
            axes[j].clear()
            axes[j].imshow(split_channels[key][0][j], cmap="gray")
            axes[j].set_title(
                f"Channel {j+1} at Time {round(key-timing_of_channels[j], 2)}"
            )

        combined_image = np.vstack(
            [split_channels[key][0][j] for j in range(number_of_channels)]
        )
        axes[number_of_channels].clear()
        axes[number_of_channels].imshow(combined_image, cmap="gray")
        axes[number_of_channels].set_title(
            f"Combined Image at Top Channel Time {round(key, 2)}"
        )

        plt.draw()

    def next_image(event):
        nonlocal index
        index = (index + 1) % len(sorted_keys)
        update_display()

    def prev_image(event):
        nonlocal index
        index = (index - 1) % len(sorted_keys)
        update_display()

    # def mark_features(event):
    #     plt.ion()
    #     key = sorted_keys[index]
    #     if key in markers:
    #         markers[key] = {}
    #     for j in range(number_of_channels):
    #         plt.figure()

    #         # Display the image
    #         plt.subplot(2, 1, 1)
    #         plt.imshow(split_channels[key][0][j], cmap="gray")
    #         plt.title(f"Channel {j+1} at Time {round(key-j, 2)}")

    #         # Calculate and display the horizontal lineout integrated over the whole height
    #         plt.subplot(2, 1, 2)
    #         horizontal_lineout = split_channels[key][0][j].sum(axis=0)
    #         plt.plot(horizontal_lineout)
    #         plt.title("Horizontal Lineout Integrated Over Height")

    #         # Get coordinates from user input
    #         coords = plt.ginput(n=1, timeout=0)

    #         plt.close()

    #         if key not in markers:
    #             markers[key] = {}
    #         markers[key][j] = coords

    def mark_features(event):
        plt.ion()
        key = sorted_keys[index]
        if key in markers:
            markers[key] = {}
        for j in range(number_of_channels):
            fig = plt.figure()

            # Set window position and size using wm_geometry
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+100+100")  # Set window position
            manager.window.wm_geometry("800x600")  # Set window size

            # Display the image
            plt.subplot(2, 1, 1)
            plt.imshow(split_channels[key][0][j], cmap="gray")
            plt.title(f"Channel {j+1} at Time {round(key-j, 2)}")

            # Calculate and display the horizontal lineout integrated over the whole height
            plt.subplot(2, 1, 2)
            horizontal_lineout = split_channels[key][0][j].sum(axis=0)
            plt.xlim(0, split_channels[key][0][j].shape[1])
            plt.plot(horizontal_lineout)
            plt.title("Horizontal Lineout Integrated Over Height")

            # Get coordinates from user input
            coords = plt.ginput(n=1, timeout=0)

            plt.close()

            if key not in markers:
                markers[key] = {}
            markers[key][j] = coords

    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.45, 0.05, 0.1, 0.075])
    axmark = plt.axes([0.6, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")
    bprev = Button(axprev, "Previous")
    bmark = Button(axmark, "Mark Features")
    bnext.on_clicked(next_image)
    bprev.on_clicked(prev_image)
    bmark.on_clicked(mark_features)

    update_display()
    plt.show()

    return markers


markers = showChannelsAndMarkFeatures(split_channels, number_of_channels)


def save_markers_to_tsv(markers, filename):
    # Create a DataFrame to store the marker coordinates
    data = []

    for key in markers:
        row = [round(key, 2)]
        for channel in range(number_of_channels):
            if channel in markers[key]:
                coords = markers[key][channel]
                if coords:
                    row.append(coords[0][0])  # X coordinate
                else:
                    row.append(None)
            else:
                row.append(None)
        data.append(row)

    # Create the DataFrame
    columns = ["time"] + [
        f"channel{channel+1}X coord" for channel in range(number_of_channels)
    ]
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a TSV file
    df.to_csv(filename, sep="\t", index=False)


def create_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.asksaveasfilename(
        defaultextension=".tsv", filetypes=[("TSV files", "*.tsv")]
    )
    if file_path:
        with open(file_path, "w") as file:
            file.write("")  # Create an empty file
        return file_path


# Save markers
filename = create_file()
print(filename)
save_markers_to_tsv(markers, filename)


# set each channel to the same contrast setting? Not sure if this is the right place to do it
# divide the before and during
# organize the images in time
