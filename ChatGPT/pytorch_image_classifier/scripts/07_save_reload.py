"""Stage 7: save weights and verify that a fresh model reproduces predictions."""
checkpoint_path = ARTIFACTS / "fashion_mnist_mlp.pt"
torch.save(model.state_dict(), checkpoint_path)
reloaded_model = ImageClassifier().to(device)
reloaded_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
reloaded_model.eval()
with torch.inference_mode():
    reloaded_logits = reloaded_model(images[:10].to(device))
assert torch.allclose(logits, reloaded_logits, atol=1e-6)
print(f"Saved {checkpoint_path}; reload verification passed.")
