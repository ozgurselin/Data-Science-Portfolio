# Script 09 - Evaluate the trained model on the held-out test set
test_accuracy = evaluate_tm(model, test_loader, accuracy).item()
print(f"Test accuracy: {test_accuracy:.4f}")
