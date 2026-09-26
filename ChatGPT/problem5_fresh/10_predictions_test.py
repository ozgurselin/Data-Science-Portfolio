model.eval()
images, labels = next(iter(valid_loader))
with torch.no_grad():
    logits = model(images[:3].to(device))
    probabilities = logits.softmax(dim=1).cpu()
predictions = logits.argmax(1).cpu()
top4_values, top4_indices = logits.topk(4, dim=1)
print("Predicted:", [classes[i] for i in predictions])
print("Actual:", [classes[i] for i in labels[:3]])
print("Full class probabilities:", probabilities.round(decimals=3))
print("Top four indices:", top4_indices.cpu())
print("Softmax conditional on top four (as in reference):", top4_values.softmax(1).cpu().round(decimals=3))
assert torch.allclose(probabilities.sum(1), torch.ones(3))
fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
for i, ax in enumerate(axes):
    ax.imshow(images[i, 0], cmap="gray")
    ax.set_title(f"Predicted: {classes[predictions[i]]}\nActual: {classes[labels[i]]}", fontsize=10)
    ax.axis("off")
fig.savefig(ARTIFACTS / "predictions.png", dpi=160)
plt.show()
plt.close(fig)
test_metrics = evaluate(model, test_loader)
assert test_metrics["count"] == 10_000
results = {"versions": versions, "seed": SEED, "epochs": n_epochs, "parameters": parameter_count,
           "training_seconds": training_seconds, "final_epoch": history[-1], "test": test_metrics,
           "predicted_labels": predictions.tolist(), "actual_labels": labels[:3].tolist()}
(ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
