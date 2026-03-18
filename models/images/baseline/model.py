# ===================== IMPORTS ======================
from tensorflow.keras import Sequential, layers, optimizers, regularizers

from tensorflow.keras.metrics import AUC

import tensorflow as tf
# ====================================================


# ═════════════════════ SIMPLE BASELINE MODEL ═════════════════════

def initialize_baseline_model():
    """
    Initialize a simple baseline CNN model for binary image classification.

    //// Architecture ////
    - Input layer: 128x128 RGB images, recaled images
    - Convolutional layers: Conv2D (kernel size, padding, strides, reLu)
        → BatchNormalization → MaxPool2D (pool size, padding)
    - Flatten → Dense (reLu) → Dropout
    - Output layer : sigmoid

    //// Return ////
    Uncompiled Keras baseline model for the compilation

    """

    baseline = Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1./255),

        layers.Conv2D(32, kernel_size=(4, 4), padding="same",
                      strides=(1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Conv2D(64, kernel_size=(3, 3), padding="same",
                      strides=(1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                      strides=(1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Conv2D(512, kernel_size=(3, 3),
                      padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    return baseline

# ░░ Compile model ░░

def compile_model(model):
    """
    Compile model for training

    //// Architecture ////
    Loss : binary crossentropy
    Optimizer : adam with a 1e-4 learning rate
    Metrics : accuracy → precision → recall → AUC

    //// Return ////
    Compiled baseline model ready for training
    """

    adam = optimizers.Adam(learning_rate=1e-4)
    model.compile(
        loss="binary_crossentropy",
        optimizer=adam,
        metrics=["accuracy", "precision", "recall", AUC(name="auc")]
    )

    return model


# ═════════════════════  BASELINE MODEL DATA AUGMENTATION ═════════════════════

def initialize_augmented_model():
    """
    Initialize CNN model with data augmentation for binary image classification.

    //// Architecture ////
    - Input layer: 128x128 RGB images, recaled images
    - Data augmentation : RandomFlip → RandomRotation → RandomZoom
    - Convolutional layers: Conv2D (kernel size, padding, strides, reLu)
        → BatchNormalization → MaxPool2D (pool size, padding)
    - GlobalAveragePooling2D → Dense (reLu, L2 regularisation) → Dropout layers
    - Output layer : sigmoid

    //// Return ////
    Uncompiled Keras model with data augmentation ready for the compilation
    """

    augmented = Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1./255),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),

        layers.Conv2D(32, kernel_size=(4, 4), padding="same",
                      strides=(1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Conv2D(64, kernel_size=(3, 3), padding="same",
                      strides=(1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                      strides=(1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.Conv2D(256, kernel_size=(3, 3),
                      padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2), padding="same"),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.Dropout(0.4),

        layers.Dense(1, activation="sigmoid")
    ])

    return augmented


# ═════════════════════  BASELINE MODEL DATA AUGMENTATION VERSION II ═════════════════════


def initialize_augmented_2_model():
    """
    Initialize CNN model with data augmentation version 2 for binary image classification.

    //// Architecture ////
    - Input layer: 128x128 RGB images, recaled images
    - Data augmentation : RandomFlip → RandomRotation
    - Convolutional layers: Conv2D (kernel size, padding, strides, reLu)
        → BatchNormalization → MaxPool2D (pool size, padding)
    - Flatten → Dense (reLu, L2 regularisation) → Dropout layers
    - Output layer : sigmoid

    //// Return ////
    Uncompiled Keras model with data augmentation version 2 ready for the compilation
    """

    augmented_2 = Sequential([
        layers.Input(shape=(128,128,3)),
        layers.Rescaling(1./255),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),

        layers.Conv2D(32, kernel_size = (3,3), padding="same", strides = (1,1), activation='relu', kernel_regularizer=regularizers.L2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), padding="same"),
        layers.Dropout(0.2),

        layers.Conv2D(64, kernel_size = (3,3), padding="same", strides = (1,1), activation='relu', kernel_regularizer=regularizers.L2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), padding="same"),
        layers.Dropout(0.2),

        layers.Conv2D(128, kernel_size = (3,3), padding="same", strides = (1,1), activation='relu', kernel_regularizer=regularizers.L2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), padding="same"),
        layers.Dropout(0.2),

        layers.Conv2D(256, kernel_size=(3,3), padding="same", activation='relu', kernel_regularizer=regularizers.L2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), padding="same"),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])
    return augmented_2
