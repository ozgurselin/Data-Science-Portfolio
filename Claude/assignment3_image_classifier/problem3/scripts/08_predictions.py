# Step 8 - Predict classes and probabilities for three validation images
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)  # index of the largest logit

print("predicted:", [class_names[i] for i in y_pred])
print("actual:   ", [class_names[i] for i in y_new[:3]])

y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("\nclass probabilities:\n", y_proba.round(decimals=3))

y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1).cpu()
print("\ntop-4 classes (probabilities renormalized over those 4):")
for i in range(3):
    print(f"  image {i}: " + ", ".join(
        f"{class_names[c]} {p:.3f}"
        for c, p in zip(y_top4_indices[i].tolist(), y_top4_probas[i].tolist())))
