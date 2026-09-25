# Step 9 - Accuracy on the 10,000-image test set
test_accuracy = evaluate_tm(model, test_loader, accuracy).item()
print(f"Test accuracy: {test_accuracy:.4f}")
