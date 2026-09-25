"""Stage 5: plot the requested training accuracy plus validation and loss."""
epochs = [row["epoch"] for row in history]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for prefix, label in [("train", "Training"), ("valid", "Validation")]:
    axes[0].plot(epochs, [100*row[f"{prefix}_accuracy"] for row in history],
                 marker="o", markersize=3, label=label)
    axes[1].plot(epochs, [row[f"{prefix}_loss"] for row in history],
                 marker="o", markersize=3, label=label)
axes[0].set(title="Training and validation accuracy", ylabel="Accuracy (%)")
axes[1].set(title="Training and validation loss", ylabel="Cross-entropy loss")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACTS / "learning_curves.png", bbox_inches="tight")
plt.show()
plt.close(fig)
print("Training accuracy is accumulated during updates; validation uses the model at each epoch's end.")
