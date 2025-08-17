import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def get_metrics(y_pred, y_true=y_true):
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics


main_metrics = get_metrics(y_test_main_model)
transfer_metrics = get_metrics(y_test_transfer_model)

# turn into dataframe
df = pd.DataFrame([
    {"name": "main model", **main_metrics},
    {"name": "transfer model", **transfer_metrics}
])

labels = ["Not AI", "AI"]

# compute matrices
confusion_matrix_main = confusion_matrix(y_true, y_test_main_model)
confusion_matrix_transfer = confusion_matrix(y_true, y_test_transfer_model)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, constrained_layout=True)

# find global max for shared colour scale
vmax = max(confusion_matrix_main.max(), confusion_matrix_transfer.max())

for ax, cm, title in zip(
    axes,
    [confusion_matrix_main, confusion_matrix_transfer],
    ["Main model", "Transfer model"]
):
    im = ax.imshow(cm, interpolation="nearest", vmin=0, vmax=vmax, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # write values inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

# shared colourbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
cbar.set_label("Counts")

plt.show()