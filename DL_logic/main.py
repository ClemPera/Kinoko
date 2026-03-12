# ===================== IMPORTS ======================
from datetime import datetime

import matplotlib as plt

from DL_logic.utils.data_utils import load_data
from DL_logic.train import train_model
from DL_logic.evaluate import prediction, evaluate
from DL_logic.utils.plots_utils import plot_baseline, plot_comparison
# ====================================================


# ░░░░░░░░░░░░░░ 🔬 Main ░░░░░░░░░░░░░░
def main():
    """
    Main entry point for the deep learning pipeline
    1. Load datasets
    2. Train baseline and
    3. Perform predictions & evaluate models on the validation sets
    4. Print results & save plots with timestamped filenames

    - 2 seconds of display for the plots before they are automatically closed.
    - Plots are stored in 'DL_logic/plots'.
    """

    # ░░ Data ░░
    train_ds, val_ds = load_data()
    class_names = train_ds.class_names
    print(train_ds.class_names)

    # ░░ Timestamp ░░
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    # ░░ Training ░░
    history_1, history_2, baseline, augmented = train_model(train_ds, val_ds, timestamp)

    # ░░ Prediction & Evaluate ░░
    prediction(baseline, augmented, val_ds, class_names, timestamp)
    evaluate(baseline, augmented, val_ds, timestamp)

    # ░░ Print Results ░░
    print("---" * 40)
    print(f"\n 🧱 Baseline")
    print(history_1.history)
    print(baseline.evaluate(val_ds))
    print("---" * 40)
    print(f"\n 💪 Augmented")
    print(history_2.history)
    print(augmented.evaluate(val_ds))
    print("---" * 40)

    # ░░ Plot Baseline ░░
    plot_baseline(history_1, timestamp)
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    # ░░ Plot Comparison ░░
    plot_comparison(history_1, history_2, timestamp)
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    print("✅ Results summary saved")


if __name__ == "__main__":
    main()
