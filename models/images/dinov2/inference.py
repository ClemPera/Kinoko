from keras import Model
from keras.utils import img_to_array
from tensorflow.data import Dataset
from PIL import ImageFile
import numpy as np


def test(model: Model, test_ds: Dataset) -> list[float]:
    """
    Evaluate a model on the test dataset and print the results.

    Args:
        model: A trained and compiled keras.Model.
        test_ds: Test tf.data.Dataset.

    Returns:
        A list of scalar metric values: [loss, accuracy, recall, precision].
    """
    results: list[float] = model.evaluate(test_ds)
    print(
        f"loss: {results[0]}\naccuracy: {results[1]}\nrecall: {results[2]}\nprecision: {results[3]}")
    return results


def predict(model: Model, image: ImageFile.ImageFile, image_sizes=(518, 518)) -> tuple[bool, float]:
    """
    Run inference on a single image and return the predicted class and confidence.

    The image is loaded, resized to (IMG_SIZE, IMG_SIZE), and passed through
    the model. The output is a sigmoid probability interpreted as:
        >= 0.5 → class 1
        <  0.5 → class 0

    Args:
        model: A trained keras.Model.
        image: Image file to classify.

    Returns:
        A tuple of (predicted_class, confidence) where predicted_class is a
        string ("0" or "1") and confidence is the raw sigmoid output in [0, 1].
    """
    img = image.resize(image_sizes)
    img_array: np.ndarray = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)

    confidence: float = float(model.predict(img_array)[0][0])
    predicted_class = confidence >= 0.5

    # Show the right confidence when it's edible
    if predicted_class is False:
        confidence = 1 - confidence

    return predicted_class, confidence
