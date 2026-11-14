
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, 
                recall_score, f1_score, roc_auc_score)
def get_losses(name, model):
    if name == "xgb":
        res = model.evals_result()
        return res["validation_0"]["logloss"], res["validation_1"]["logloss"]
    if name == "lgb":
        res = model.evals_result_
        keys = list(res.keys())
        return res[keys[0]]["binary_logloss"], res[keys[1]]["binary_logloss"]
    if name == "cb":
        res = model.get_evals_result()
        keys = list(res.keys())
        return res[keys[0]]["Logloss"], res[keys[1]]["Logloss"]

def calc_loss(model_dict):   
    train_loss, val_loss = {}, {}

    for name, model in model_dict.items():
        train_loss[name], val_loss[name] = get_losses(name, model)

    return train_loss, val_loss

def plot(train_loss, val_loss):    
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    colors = {"XGBoost": "#1f4e79", "LightGBM": "#2e7d32", "CatBoost": "#b03a2e"}
    curves = {
        "XGBoost":  (train_loss["xgb"], val_loss["xgb"]),
        "LightGBM": (train_loss["lgb"], val_loss["lgb"]),
        "CatBoost": (train_loss["cb"],  val_loss["cb"]),
    }
    label_dy = {"XGBoost": 10, "LightGBM": -10, "CatBoost": 0}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    gaps = {}
    for name, (train, val) in curves.items():
        c = colors[name]
        best = val.index(min(val)) + 1
        ax1.plot(range(1, len(val) + 1), val, color=c, linewidth=2.6)
        ax1.scatter(best, min(val), color=c, edgecolor="white", linewidth=1.5, s=100, zorder=4)

        end_x = min(len(val), 600)
        txt = ax1.annotate(f"{name}  {min(val):.3f}", xy=(end_x, val[end_x - 1]),
                        xytext=(10, label_dy[name]), textcoords="offset points",
                        color=c, fontsize=13, fontweight="bold", va="center")
        # white outline keeps text readable where it crosses gridlines or lines
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        gaps[name] = val[best - 1] - train[best - 1]

    ax1.set_xlim(0, 820)
    ax1.set_ylim(0.40, 0.60)
    ax1.set_xlabel("Boosting rounds (trees)", fontsize=13)
    ax1.set_ylabel("Validation log loss (lower is better)", fontsize=13)
    ax1.set_title("Validation loss, first 600 trees", fontsize=14, fontweight="bold", loc="left")

    names = list(gaps)
    bars = ax2.barh(names, [gaps[n] for n in names], color=[colors[n] for n in names], height=0.55)
    for bar, n in zip(bars, names):
        ax2.text(bar.get_width() + 0.0015, bar.get_y() + bar.get_height() / 2,
                f"{gaps[n]:.3f}", va="center", fontsize=13, fontweight="bold", color="#111111")
    ax2.set_xlim(0, max(gaps.values()) * 1.3)
    ax2.invert_yaxis()
    ax2.set_xlabel("Validation minus training loss", fontsize=13)
    ax2.set_title("Overfitting gap at best tree", fontsize=14, fontweight="bold", loc="left")
    ax2.tick_params(axis="y", labelsize=13)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y" if ax is ax1 else "x", alpha=0.2)

    fig.suptitle("Customer Churn Model: Log Loss by Gradient Boosting Library",
                fontsize=17, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.02,
            "Each model uses its own tuned hyperparameters, so tree counts are not a speed comparison. "
            "Dots mark each model's lowest validation loss.",
            fontsize=11, color="#222222", ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("outputs/figures/all_libraries_loss.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

def evaluate(model, x_test, x_test_unenc, y_test):
    if model.__class__.__name__ == "CatBoostClassifier":
        x_test = x_test_unenc  # CatBoost can handle categorical features directly

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1": f"{f1:.4f}",
        "roc_auc": f"{roc_auc:.4f}"
    }

def main():
    print()
    print()
    print('Evaluating the models...')
    print()    
    xgb=joblib.load('models/xgb_model.joblib')
    lgb=joblib.load('models/lgb_model.joblib')
    cb=joblib.load('models/cb_model.joblib')

    print('  Loaded the trained models for evaluation.')
    print()

    y_test=pd.read_csv('data/processed/y_test.csv')
    x_test=pd.read_csv('data/processed/test_enc.csv')
    x_test_unenc=pd.read_csv('data/processed/x_test.csv')
    print('  Loaded the test data for evaluation.')
    print()

    eval_xgb=evaluate(xgb, x_test, x_test_unenc, y_test)
    print('  Evaluated the XGBoost model on the test set.')
    print(f"  XGBoost Test Metrics: {eval_xgb}")
    eval_lgb=evaluate(lgb, x_test, x_test_unenc, y_test)
    print('  Evaluated the LightGBM model on the test set.')
    print(f"  LightGBM Test Metrics: {eval_lgb}")
    eval_cb=evaluate(cb, x_test, x_test_unenc, y_test)
    print('  Evaluated the CatBoost model on the test set.')
    print(f"  CatBoost Test Metrics: {eval_cb}")
    print()
    print(pd.DataFrame({"XGBoost": eval_xgb, "LightGBM": eval_lgb, "CatBoost": eval_cb}).T)
    print()

    model_dict = {'xgb': xgb, 'lgb': lgb, 'cb': cb}
    train_loss, val_loss = calc_loss(model_dict)
    print('  Calculated training and validation loss for each model.')
    print()

    plot(train_loss, val_loss)
    print('  Plotted the training and validation loss curves.')
    print('  Saved the plot to outputs/figures/all_libraries_loss.png')
    print('Done!')

if __name__ == "__main__":
    main()

