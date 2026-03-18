# ===================== IMPORTS ======================
import os
import pandas as pd
import datetime

# ====================================================

# ===================== 🔮 Predictions ======================


def prediction(model, image, save_result: bool = False) -> tuple[bool, float]:
    """
    Predict probabilities and labels for both models
    """

    # ░░ Probabilities ░░
    # preds_base = probability that the image belongs to class 1
    preds_base = float(model.predict(image, verbose=0)[0][0])

    # ░░ Labels ░░
    # Convert baseline predicted probabilities into binary class labels using a 0.5 threshold.
    pred_class = preds_base >= 0.5

    # Show the right confidence when it's edible
    if pred_class == False:
        preds_base = 1 - preds_base

    if save_result:
        os.makedirs("../models/images/baseline/results", exist_ok=True)
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        df_proba_results = pd.DataFrame({
            "Probability": preds_base,
            "Predicted_class": pred_class,
        })
        df_proba_results.to_csv(
            f"../models/images/baseline/results/baselines_probability_{timestamp}.csv", index=False)

    return pred_class, preds_base

# ===================== 🏁 Evaluate ======================

def evaluate(baseline, augmented, augmented_2, val_ds, timestamp: str):
    """
    Evaluate both models performance and save results
    """

    baseline_eval = baseline.evaluate(val_ds, return_dict=True)
    augmented_eval = augmented.evaluate(val_ds, return_dict=True)
    augmented_2_eval = augmented_2.evaluate(val_ds, return_dict=True)

    results = [
        {
            "model": "🧱 baseline",
            **baseline_eval
        },
        {
            "model": "💪 augmented",
            **augmented_eval
        },
        {
            "model": "🦾 augmented_2",
            **augmented_2_eval
        }
    ]

    df_results = pd.DataFrame(results)

    os.makedirs("../models/images/baseline/results", exist_ok=True)
    df_results.to_csv(
        f"../models/images/baseline/results/df_results_eval_{timestamp}.csv", index=False)

    return df_results
