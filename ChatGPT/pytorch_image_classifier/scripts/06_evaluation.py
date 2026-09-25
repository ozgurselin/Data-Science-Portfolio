"""Stage 6: evaluate the validation-selected model on the untouched test set."""
model.load_state_dict(best_state)
test_loss, test_accuracy = evaluate(model, test_loader)
print(f"Selected epoch: {best_epoch}; test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.2%}")
images, targets = next(iter(test_loader))
model.eval()
with torch.inference_mode():
    logits = model(images[:10].to(device))
    probabilities = logits.softmax(dim=1).cpu()
    predictions = probabilities.argmax(dim=1)
assert torch.allclose(probabilities.sum(dim=1), torch.ones(10), atol=1e-6)
top4_probabilities, top4_indices = probabilities[:3].topk(4, dim=1)
for index in range(3):
    print(f"Example {index + 1}: true={class_names[targets[index]]}; "
          f"predicted={class_names[predictions[index]]}")
    print([(class_names[c], round(float(p), 4))
           for p, c in zip(top4_probabilities[index], top4_indices[index])])
fig, axes = plt.subplots(2, 5, figsize=(13, 5))
for index, ax in enumerate(axes.flat):
    ax.imshow(images[index].squeeze().numpy(), cmap="gray")
    correct = predictions[index] == targets[index]
    ax.set_title(f"True: {class_names[targets[index]]}\nPred: {class_names[predictions[index]]}",
                 fontsize=9, color="darkgreen" if correct else "firebrick")
    ax.axis("off")
fig.suptitle("Test predictions from the validation-selected model")
fig.tight_layout()
fig.savefig(ARTIFACTS / "test_predictions.png", bbox_inches="tight")
plt.show()
plt.close(fig)
results = {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
           "learning_rate": LEARNING_RATE, "best_epoch": best_epoch,
           "best_validation_accuracy": best_valid_accuracy,
           "final_training_accuracy": history[-1]["train_accuracy"],
           "test_loss": test_loss, "test_accuracy": test_accuracy,
           "training_seconds": training_seconds, "parameter_count": parameter_count,
           "python": platform.python_version(), "torch": str(torch.__version__),
           "torchvision": str(torchvision.__version__), "device": str(device)}
(ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2) + "\n")
