import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Directory containing the images
trial = 46
trialsDirectory = "2025-04-14"
imageDirectory = os.path.join(trialsDirectory,str(trial))

# Set trials directory path
trialsDirectory = r"..\2025-04-16"

# List all subdirectories (i.e., folders) in the trials directory
subdirectories = [f for f in os.listdir(trialsDirectory) if os.path.isdir(os.path.join(trialsDirectory, f))]

# Filter the subdirectories to keep only those that are numeric (i.e., trial numbers)
numeric_folders = [int(f) for f in subdirectories if f.isdigit()]

# Find the highest trial number
trial = np.argmax(numeric_folders)


# Set the image directory path to the folder with the highest trial number
imageDirectory = os.path.join(trialsDirectory, subdirectories[trial])
print(imageDirectory)
# Background image
bkgImagePump = r"C:\Users\twardowski.6a\Documents\SEProbe\2025-04-13\bkg pump only.tif"
bkgImageNone = r"C:\Users\twardowski.6a\Documents\SEProbe\2025-04-13\bkg both blocked.tif"
# Function to sort images by date (assuming the date is part of the filename in the format 'YYYY-MM-DD')
def load_images_by_date(directory):
    # List all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(('.tif', '.jpg', '.png'))]
    
    # Sort the files by the date in the filename
    image_files.sort()  # This works if filenames include dates in a sortable format (e.g., "2025-04-07")
    
    # Load the images into a dictionary
    images = {}
    for file in image_files:
        image_path = os.path.join(directory, file)
        images[file] = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    
    return images

# Load images sorted by date
images = load_images_by_date(imageDirectory)

# Assuming the first image is "before" and the last image is "during"
image_filenames = list(images.keys())
before_image = images[image_filenames[0]]
during_image = images[image_filenames[1]]




# Optionally, you can load the background image as well
backgroundPump = cv2.imread(bkgImagePump, cv2.IMREAD_ANYDEPTH).astype(np.float32)
backgroundNone = cv2.imread(bkgImageNone, cv2.IMREAD_ANYDEPTH).astype(np.float32)

#during_image = during_image-backgroundPump
#before_image = before_image-backgroundNone

before_image = np.clip(before_image,1,65535)
during_image = np.clip(during_image,1,65535)
#compositeImage = (during_image-background)/(before_image-background)
compositeImage = during_image/(before_image)
compositeImage = np.clip(compositeImage, 0.001, 65535)


# Display the images using matplotlib
plt.subplot(1, 3, 1)
plt.imshow(before_image, cmap='gray')
plt.title("Before Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(during_image, cmap='gray')
plt.title("During Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(backgroundPump, cmap='gray')
plt.title("Background Image")
plt.axis('off')

plt.show()

# Display the composite image
print(np.max(compositeImage))
plt.imshow(compositeImage, cmap='gray', vmin=np.min(compositeImage))
plt.title("Composite Image (Full Precision)")
plt.axis('off')
plt.show()