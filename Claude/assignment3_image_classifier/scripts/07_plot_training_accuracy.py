# Script 07 - NEW: plot the training accuracy (with validation for reference)
# The training accuracy is averaged over each epoch while the model is still
# learning, so it is plotted half an epoch earlier (same convention as the book).
epochs = np.arange(1, n_epochs + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs - 0.5, history["train_metrics"], ".--", label="Training")
ax.plot(epochs, history["valid_metrics"], ".-", label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Fashion MNIST classifier - training accuracy")
ax.set_xlim(0, n_epochs + 0.5)
ax.grid()
ax.legend()
plt.show()

print(f"Final training accuracy:   {history['train_metrics'][-1]:.4f}")
print(f"Final validation accuracy: {history['valid_metrics'][-1]:.4f}")
