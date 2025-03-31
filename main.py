import os
import pandas as pd
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button


# User Inputs
directory = "data/TA/455mw"
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
    time = zero_time - float(df.loc[df["trial"] == trial, "time"].values)
    # Store images in the dictionaries
    if time not in before_images_by_time:
        before_images_by_time[time] = [before_image]
        during_images_by_time[time] = [during_image]
        normalized_images_by_time[time] = [normalized_image]
        edge_positions[time] = [edge_position]
    else:
        before_images_by_time[time].append(before_image)
        during_images_by_time[time].append(during_image)
        normalized_images_by_time[time].append(normalized_image)
        edge_positions[time].append(edge_position)


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


showImages(before_images_by_time, draw_line=False)
showImages(during_images_by_time, draw_line=False)
showImages(normalized_images_by_time, draw_line=False)


# Crop all images to each channel

# set each channel to the same contrast setting? Not sure if this is the right place to do it
# divide the before and during
# organize the images in time
