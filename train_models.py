"""
Thyroid Disease Binary Classification Pipeline
==============================================
This script loads the pre-balanced training dataset and untouched test dataset,
constructs a leak-free ColumnTransformer & Pipeline, trains Logistic Regression,
Random Forest, and Gradient Boosting classifiers, evaluates all models strictly
on the test set, generates diagnostic visualizations, and exports the final
results table and best model to outputs/.

Clinical Context:
-----------------
In health screening, missing an actual disease case (False Negative) can have
fatal or life-altering consequences (e.g. unmanaged hypothyroidism leading to
cardiovascular complications, metabolic failure, or myxedema coma). Conversely,
a False Positive merely results in confirmatory repeat blood tests.
Because the test set reflects real-world class imbalance (~74% healthy, ~26% condition),
a naive classifier predicting 0 for all patients would obtain 73.84% accuracy while
detecting 0% of sick patients. Therefore, Recall on the positive class (sensitivity)
is the primary optimization and ranking metric.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Windows Application Control can block HistGradientBoosting DLLs.
# Stub that submodule so RandomForest and GradientBoosting still import.
import sys
from types import ModuleType
if "sklearn.ensemble._hist_gradient_boosting.gradient_boosting" not in sys.modules:
    _hist_pkg = ModuleType("sklearn.ensemble._hist_gradient_boosting")
    _hist_mod = ModuleType("sklearn.ensemble._hist_gradient_boosting.gradient_boosting")
    _hist_mod.HistGradientBoostingClassifier = object
    _hist_mod.HistGradientBoostingRegressor = object
    sys.modules["sklearn.ensemble._hist_gradient_boosting"] = _hist_pkg
    sys.modules["sklearn.ensemble._hist_gradient_boosting.gradient_boosting"] = _hist_mod

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


def choose_threshold(y_true, proba, min_precision=0.80):
    """Pick a decision threshold on a validation split.

    Prefer the lowest threshold that keeps precision at least min_precision
    and maximises recall (screening: miss fewer true cases). Fall back to
    the best F1 if no threshold meets the precision floor.
    """
    best_valid = None
    best_f1 = None
    for threshold in np.linspace(0.20, 0.55, 36):
        pred = (proba >= threshold).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        row = {"threshold": float(threshold), "precision": prec, "recall": rec, "f1": f1}
        if best_f1 is None or f1 > best_f1["f1"]:
            best_f1 = row
        if prec >= min_precision and (best_valid is None or rec > best_valid["recall"]):
            best_valid = row
    return best_valid or best_f1

def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Paths & Setup
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / "thyroid_train_balanced.csv"
    test_path = out_dir / "thyroid_test.csv"
    
    print(f"Loading training data from: {train_path.resolve()}")
    print(f"Loading test data from:     {test_path.resolve()}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    target_col = "has_thyroid_condition"
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features (Balanced: {y_train.value_counts().to_dict()})")
    print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features (Real-world: {y_test.value_counts().to_dict()})")
    
    # -------------------------------------------------------------------------
    # 2. Build Preprocessing Pipeline (Fitted on Training Data Only)
    # -------------------------------------------------------------------------
    numeric_features = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    binary_features = [col for col in X_train.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("bin", "passthrough", binary_features)
        ],
        remainder="drop"
    )
    
    # -------------------------------------------------------------------------
    # 3. Define Classifiers
    # -------------------------------------------------------------------------
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                C=0.8,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=400,
                max_depth=16,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )),
        ]),
    }
    
    # -------------------------------------------------------------------------
    # 4. Train and Evaluate Strictly on Test Set
    # -------------------------------------------------------------------------
    evaluation_records = []
    test_probs = {}
    test_preds = {}
    fitted_models = {}
    cms = {}
    
    cleaned_path = out_dir / "thyroid_cleaned.csv"
    if cleaned_path.exists():
        cleaned = pd.read_csv(cleaned_path)
        leftover = pd.concat([cleaned, df_test, df_test]).drop_duplicates(keep=False)
        X_left = leftover.drop(columns=[target_col])
        y_left = leftover[target_col]
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_left, y_left, test_size=0.25, stratify=y_left, random_state=42
        )
        fit_df = X_fit.copy()
        fit_df[target_col] = y_fit.values
        majority = fit_df[fit_df[target_col] == 0]
        minority = fit_df[fit_df[target_col] == 1]
        minority_up = minority.sample(n=len(majority), replace=True, random_state=42)
        fit_bal = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
        X_fit = fit_bal.drop(columns=[target_col])
        y_fit = fit_bal[target_col]
        print(
            f"Threshold validation: leftover original patients "
            f"(val n={len(y_val)}, positive rate={y_val.mean():.3f})"
        )
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

    for name, pipe in models.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_fit, y_fit)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        chosen = choose_threshold(y_val, val_prob, min_precision=0.80)
        threshold = chosen["threshold"]
        print(
            f"  Validation threshold={threshold:.2f} "
            f"(precision={chosen['precision']:.3f}, recall={chosen['recall']:.3f})"
        )

        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        test_preds[name] = y_pred
        test_probs[name] = y_prob
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cms[name] = cm
        
        evaluation_records.append({
            "Model": name,
            "Threshold": threshold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall (Positive)": rec,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "False Negatives (FN)": int(cm[1, 0]),
            "False Positives (FP)": int(cm[0, 1])
        })
        
        print(f"=== {name} Confusion Matrix ===")
        print(f"  TN: {cm[0, 0]:4d} | FP: {cm[0, 1]:4d}")
        print(f"  FN: {cm[1, 0]:4d} | TP: {cm[1, 1]:4d}")
        print(f"  -> Recall on Positive Class: {rec:.4f} ({cm[1, 1]}/{cm[1, 0] + cm[1, 1]} detected)")
        print(f"  -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    for name, pipe in fitted_models.items():
        y_prob = test_probs[name]
        y_pred_default = (y_prob >= 0.50).astype(int)
        cm_default = confusion_matrix(y_test, y_pred_default)
        evaluation_records.append({
            "Model": f"{name} (threshold 0.50)",
            "Threshold": 0.50,
            "Accuracy": accuracy_score(y_test, y_pred_default),
            "Precision": precision_score(y_test, y_pred_default),
            "Recall (Positive)": recall_score(y_test, y_pred_default),
            "F1 Score": f1_score(y_test, y_pred_default),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "False Negatives (FN)": int(cm_default[1, 0]),
            "False Positives (FP)": int(cm_default[0, 1]),
        })

    # -------------------------------------------------------------------------
    # 5. Results Table (Sorted by Recall on Positive Class)
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(evaluation_records).drop_duplicates()
    results_df = results_df.sort_values(by="Recall (Positive)", ascending=False).reset_index(drop=True)
    
    results_csv_path = out_dir / "model_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n=======================================================")
    print(f"Final Model Comparison (Sorted by Recall on Positive):")
    print(f"=======================================================")
    print(results_df.to_string(index=False))
    print(f"Saved results table to {results_csv_path.resolve()}")
    
    # -------------------------------------------------------------------------
    # 6. Confusion Matrices Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Condition (0)", "Condition (1)"],
            yticklabels=["No Condition (0)", "Condition (1)"],
            ax=ax, annot_kws={"size": 13, "weight": "bold"}
        )
        fn_count = cm[1, 0]
        ax.set_title(f"{name}\nFalse Negatives: {fn_count}", fontsize=12, pad=10, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    plt.suptitle("Confusion Matrices (Prioritising Minimum False Negatives)", fontsize=14, y=1.03, weight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 7. Plot Shared ROC Curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8.5, 6.5))
    colors = {
        "Gradient Boosting": "#1f77b4",
        "Random Forest": "#2ca02c",
        "Logistic Regression": "#ff7f0e"
    }
    for name in ["Gradient Boosting", "Random Forest", "Logistic Regression"]:
        y_prob = test_probs[name]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors[name], linewidth=2.2)
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance (AUC = 0.5000)", alpha=0.7)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, pad=12, weight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    roc_plot_path = out_dir / "roc_curves_comparison.png"
    plt.savefig(roc_plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved ROC curves plot to {roc_plot_path.resolve()}")
    
    # -------------------------------------------------------------------------
    # 8. Feature Importances / Coefficients Analysis
    # -------------------------------------------------------------------------
    feature_names = numeric_features + binary_features
    
    # Gradient Boosting Importances
    gb_importances = pd.Series(
        fitted_models["Gradient Boosting"].named_steps["classifier"].feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    # Random Forest Importances
    rf_importances = pd.Series(
        fitted_models["Random Forest"].named_steps["classifier"].feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    # Logistic Regression Coefficients
    lr_coefs = fitted_models["Logistic Regression"].named_steps["classifier"].coef_[0]
    lr_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lr_coefs,
        "Odds_Ratio": np.exp(lr_coefs)
    }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    gb_importances.head(10).sort_values().plot(
        kind="barh", ax=axes[0], color="#1f77b4", edgecolor="black", alpha=0.85
    )
    axes[0].set_title("Gradient Boosting: Top 10 Feature Importances", fontsize=12, weight="bold")
    axes[0].set_xlabel("Relative Importance (Gini)", fontsize=10)
    
    rf_importances.head(10).sort_values().plot(
        kind="barh", ax=axes[1], color="#2ca02c", edgecolor="black", alpha=0.85
    )
    axes[1].set_title("Random Forest: Top 10 Feature Importances", fontsize=12, weight="bold")
    axes[1].set_xlabel("Relative Importance (MDI)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importances.png", bbox_inches="tight")
    plt.close()
    print(f"Saved feature importances plot to {(out_dir / 'feature_importances.png').resolve()}")
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "random_forest": rf_importances.reindex(feature_names).values,
        "gradient_boosting": gb_importances.reindex(feature_names).values,
    }).sort_values("random_forest", ascending=False)
    importance_path = out_dir / "feature_importances.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importances to {importance_path.resolve()}")

    print("\nTop 5 Drivers for Gradient Boosting:")
    print(gb_importances.head(5).to_string())
    
    print("\nLogistic Regression Top Positive Risk Drivers (Odds Ratio > 1):")
    print(lr_df.head(5).to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 9. Save Best Model
    # -------------------------------------------------------------------------
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    best_model_path = out_dir / "best_thyroid_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"\nBest Model: '{best_model_name}' (Highest Recall: {results_df.iloc[0]['Recall (Positive)']:.4f})")
    print(f"Saved best model pipeline to: {best_model_path.resolve()}")

if __name__ == "__main__":
    run_pipeline()
