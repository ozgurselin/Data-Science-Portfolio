# Script 03 - Create DataLoaders and inspect one sample
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Each entry is a tuple (image, target); images are [channels, rows, columns]
X_sample, y_sample = train_data[0]
print("shape:", X_sample.shape, "| dtype:", X_sample.dtype,
      "| class:", train_and_valid_data.classes[y_sample])
