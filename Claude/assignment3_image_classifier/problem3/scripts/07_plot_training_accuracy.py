# Step 7 (new) - Plot the training accuracy (with validation accuracy and loss)
epochs = np.arange(1, n_epochs + 1)
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

# Training metrics are averaged during each epoch, so shift them back half an epoch
ax_acc.plot(epochs - 0.5, history["train_metrics"], ".--", label="Training")
ax_acc.plot(epochs, history["valid_metrics"], ".-", label="Validation")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Training accuracy")
ax_acc.set_xlim(0, n_epochs + 0.5)
ax_acc.grid()
ax_acc.legend()

ax_loss.plot(epochs - 0.5, history["train_losses"], ".--", color="C0")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("Training loss")
ax_loss.set_xlim(0, n_epochs + 0.5)
ax_loss.grid()

plt.tight_layout()
plt.show()

best = int(np.argmax(history["valid_metrics"]))
print(f"Final training accuracy:   {history['train_metrics'][-1]:.4f}")
print(f"Final validation accuracy: {history['valid_metrics'][-1]:.4f} "
      f"(best {history['valid_metrics'][best]:.4f} at epoch {best + 1})")
