# ===================== IMPORTS ======================
from datetime import datetime

from data import load_data
from train import train_model
from evaluate import prediction, evaluate
from utils import plot_baseline, plot_comparison
# ====================================================


# ░░░░░░░░░░░░░░ 🔬 Main ░░░░░░░░░░░░░░
def main():

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

    plot_baseline(history_1, timestamp)
    plot_comparison(history_1, history_2, timestamp)

    print("✅ Results summary saved")


if __name__ == "__main__":
    main()
