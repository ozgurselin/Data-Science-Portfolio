# Script 08 - Predict classes and class probabilities for new images
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)  # index of the largest logit

print("predicted:", [train_and_valid_data.classes[index] for index in y_pred])
print("actual:   ", [train_and_valid_data.classes[index] for index in y_new[:3]])

y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("probabilities:\n", y_proba.round(decimals=3))

y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1).cpu()
print("top-4 probabilities:\n", y_top4_probas.round(decimals=3))
print("top-4 classes:\n", y_top4_indices.cpu())
