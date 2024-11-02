import numpy as np
import gzip


def load_dataset(
    features_path: str, 
    labels_path: str,
    total_images: int,
    image_size: int = 28
):
    with gzip.open(features_path, "r") as f:
        # the first 16 bytes is the header, .read(16) effectively skips it
        f.read(16) 

        # defining how to read the data 
        buf = f.read(image_size * image_size * total_images)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        
        # the dimensions here are should actually be 
        # (number of images, channels, image height, image width)
        # but I am loading it for plotting the image using numpy 
        # and I reshaped it later for torch
        images = images.reshape(total_images, image_size, image_size, 1)

    # load the labels
    with gzip.open(labels_path, "r") as f:
        # the first 8 bytes is the header, skipping it
        f.read(8)
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int_)

    return images, labels
