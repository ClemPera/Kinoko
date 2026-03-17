from keras import Input, layers, KerasTensor
from keras_hub.models import DINOV2Backbone


def get_cls_dinov2(img_shape=(512, 512, 3)) -> tuple[KerasTensor, KerasTensor]:
    """
    Build a frozen DINOv2 backbone and return its CLS token output.

    The backbone is loaded from the "dinov2_base" preset and frozen.
    Input images are rescaled to [0, 1] and normalized with ImageNet statistics
    before being passed to the backbone.

    Returns:
        - A KerasTensor of shape (batch, hidden_dim) representing the CLS token
            extracted from the last layer of the backbone.
        - A Kera Input reprensenting the inputs used by the model
    """
    backbone: DINOV2Backbone = DINOV2Backbone.from_preset("dinov2_base")
    backbone.trainable = False

    inputs: KerasTensor = Input(shape=img_shape)  # type: ignore
    x: KerasTensor = layers.Rescaling(1 / 255.0)(inputs)
    x = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2]
    )(x)

    backbone_out: KerasTensor = backbone({"images": x})
    outputs: KerasTensor = backbone_out[:, 0, :]

    return inputs, outputs
