import numpy as np
from tensorflow.data import Dataset
from PIL import ImageFile

def pil_to_dataset(img: ImageFile.ImageFile, img_size=(128, 128)) -> Dataset:
    '''
    Resize and convert to array
    Args:
        - img: ImageFile of the image
        - img_size: tuple specifyning which size the image should be resized
    Returns: 
        The Dataset of the image
    '''
    # Resize and convert to array
    img = img.resize(img_size)
    img_array = np.array(img)  # shape: (H, W, 3)
    
    # Add batch dimension → (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Wrap in a tf.data.Dataset
    ds = Dataset.from_tensors(img_array)
    
    return ds