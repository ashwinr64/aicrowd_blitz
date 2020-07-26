import os
import glob
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed
import cv2
import numpy as np

# Set white bg
def white_bg(image_path, output_path, size):
    base_name = os.path.basename(image_path)
    output = os.path.join(output_path, base_name)

    # Read with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Isolate tranparent region
    trans = np.where(img[:, :, 3] == 128)
    skel = np.where(img[:, :, 3] != 128)

    # Replace transparent region with white and skeleton as black
    img[trans] = (255, 255, 255, 0)
    img[skel] = (0, 0, 0, 0)

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, size)

    # Save
    cv2.imwrite(output, img)


size = (224, 224)
image_path = '../data_original/training/images/'
output_path = '../data/training_masked/'
images = glob.glob(os.path.join(image_path, '*'))

Parallel(n_jobs=12)(
    delayed(white_bg)(
        i,
        output_path,
        size
    ) for i in tqdm(images)
)
