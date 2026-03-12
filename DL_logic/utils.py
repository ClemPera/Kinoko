
# ===================== IMPORTS ======================
import os

import matplotlib.pyplot as plt

# ====================================================


# ░░░░░░░░░░░░░░ 📈 Plots : Baseline ░░░░░░░░░░░░░░
def plot_baseline(history, timestamp):
    """
    Plot training metrics (Loss, Accuracy, Precision, Recall, AUC)
    for the baseline model.
    """

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    # ░░ Row 1 : Train & Val Loss ░░
    ax[0].set_title('Loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Val loss")

    # ░░ Row 1 : Train & Val Accuracy ░░
    ax[1].set_title('Accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Val acc")

    # ░░ Row 1 : Train & Val Precision ░░
    ax[2].set_title('Precision')
    ax[2].plot(history.epoch, history.history["precision"], label="Train precision")
    ax[2].plot(history.epoch, history.history["val_precision"], label="Val precision")

    # ░░ Row 1 : Train & Val Recall ░░
    ax[3].set_title('Recall')
    ax[3].plot(history.epoch, history.history["recall"], label="Train recall")
    ax[3].plot(history.epoch, history.history["val_recall"], label="Val recall")

    # ░░ Row 1 : Train & Val AUC ░░
    ax[4].set_title('AUC')
    ax[4].plot(history.epoch, history.history["auc"], label="Train AUC")
    ax[4].plot(history.epoch, history.history["val_auc"], label="Val AUC")

    for a in ax:
        a.legend()

    plt.tight_layout()

    # ░░ 💾 Save ░░
    os.makedirs("plots/DL_plots", exist_ok=True)
    plt.savefig(f"plots/DL_plots/baseline_{timestamp}.png")

    plt.show()

# ░░░░░░░░░░░░░░ 📈 Plots : Baseline vs Augmented ░░░░░░░░░░░░░░
def plot_comparison(history, history_2, timestamp):
    """
    Compare training metrics (Loss, Accuracy, Precision, Recall, AUC)
    for the two models.
    """

    fig, ax = plt.subplots(6, 1, figsize=(12, 24))

    # ░░ Row 1 : Train loss ░░
    ax[0].set_title('Train Loss Comparison')
    ax[0].plot(history.epoch, history.history["loss"], label="Baseline")
    ax[0].plot(history_2.epoch, history_2.history["loss"], label="Augmented", color="red")
    ax[0].legend()

    # ░░ Row 2 : Val loss ░░
    ax[1].set_title('Validation Loss Comparison')
    ax[1].plot(history.epoch, history.history["val_loss"], label="Baseline")
    ax[1].plot(history_2.epoch, history_2.history["val_loss"], label="Augmented", color="red")
    ax[1].legend()

    # ░░ Row 3 : Train Accuracy ░░
    ax[2].set_title('Train Accuracy Comparison')
    ax[2].plot(history.epoch, history.history["accuracy"], label="Baseline")
    ax[2].plot(history_2.epoch, history_2.history["accuracy"], label="Augmented", color="red")
    ax[2].legend()

    # ░░ Row 4 : Val Accuracy ░░
    ax[3].set_title('Validation Accuracy Comparison')
    ax[3].plot(history.epoch, history.history["val_accuracy"], label="Baseline")
    ax[3].plot(history_2.epoch, history_2.history["val_accuracy"], label="Augmented", color="red")
    ax[3].legend()

    # ░░ Row 5 : Train AUC ░░
    ax[4].set_title('Train AUC Comparison')
    ax[4].plot(history.epoch, history.history["auc"], label="Baseline")
    ax[4].plot(history_2.epoch, history_2.history["auc"], label="Augmented", color="red")
    ax[4].legend()

    # ░░ Row 6 : Val AUC ░░
    ax[5].set_title('Validation AUC Comparison')
    ax[5].plot(history.epoch, history.history["val_auc"], label="Baseline")
    ax[5].plot(history_2.epoch, history_2.history["val_auc"], label="Augmented", color="red")
    ax[5].legend()

    for row in ax:
        row.legend()

    plt.tight_layout()

    # ░░ 💾 Save ░░

    os.makedirs("plots/DL_plots", exist_ok=True)
    plt.savefig(f"plots/DL_plots/comparison_{timestamp}.png")

    plt.show()
