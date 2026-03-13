# ===================== IMPORTS ======================
import os
import pandas as pd

# ====================================================

# ===================== 🔮 Predictions ======================
def prediction(baseline, augmented, val_ds, class_names, timestamp):
    """
    Predict probabilities and labels for both models
    """

    # ░░ Probabilities ░░
    # preds_base = probability that the image belongs to class 1
    preds_base = baseline.predict(val_ds, verbose=0)
    # preds_aug = probability that the image belongs to class 1
    preds_augm = augmented.predict(val_ds, verbose=0)

    # ░░ Labels ░░
    # Convert baseline predicted probabilities into binary class labels using a 0.5 threshold.
    pred_labels = (preds_base > 0.5).astype(int)
    # Convert augmented baseline predicted probabilities into binary class labels using a 0.5 threshold.
    pred_labels_2 = (preds_augm > 0.5).astype(int)

    # ░░ DataFrame ░░

    df_proba_results = pd.DataFrame({
        "Baseline_proba": preds_base.flatten(),
        "Augmented_proba": preds_augm.flatten(),
        "Baseline_label": pred_labels.flatten(),
        "Augmented_label": pred_labels_2.flatten()
    })

    class_map = dict(enumerate(class_names))
    df_proba_results["Baseline_class"] = df_proba_results["Baseline_label"].map(class_map)
    df_proba_results["Augmented_class"] = df_proba_results["Augmented_label"].map(class_map)

    os.makedirs("DL_logic/results", exist_ok=True)
    df_proba_results.to_csv(f"DL_logic/results/baselines_probability_{timestamp}.csv", index=False)

    return df_proba_results

# ===================== 🏁 Evaluate ======================

def evaluate(baseline, augmented, val_ds, timestamp):
    """
    Evaluate both models performance and save results
    """

    baseline_eval = baseline.evaluate(val_ds, return_dict=True)
    augmented_eval = augmented.evaluate(val_ds, return_dict=True)

    results = [
        {
        "model": "🧱 baseline",
        **baseline_eval
    },
    {
        "model": "💪 augmented",
        **augmented_eval
    }
    ]

    df_results = pd.DataFrame(results)

    os.makedirs("DL_logic/results", exist_ok=True)
    df_results.to_csv(f"DL_logic/results/df_results_eval_{timestamp}.csv", index=False)

    return df_results
