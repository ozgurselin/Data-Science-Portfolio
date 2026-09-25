# Step 3 - Data loaders and a look at the samples
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

X_sample, y_sample = train_data[0]  # each entry is an (image, target) tuple
print("shape:", X_sample.shape, "| dtype:", X_sample.dtype,
      "| label:", class_names[y_sample])

fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for ax, idx in zip(axes.flat, range(16)):
    image, label = train_data[idx]
    ax.imshow(image.squeeze(), cmap="binary")
    ax.set_title(class_names[label], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
