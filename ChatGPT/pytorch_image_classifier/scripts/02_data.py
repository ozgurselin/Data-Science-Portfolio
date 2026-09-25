"""Stage 2: download Fashion-MNIST and create disjoint data loaders."""
to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=to_tensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=to_tensor)
train_data, valid_data = random_split(
    train_and_valid_data, [55_000, 5_000],
    generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED), num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)
class_names = train_and_valid_data.classes
assert set(train_data.indices).isdisjoint(valid_data.indices)
image, label = train_data[0]
assert image.shape == (1, 28, 28) and image.dtype == torch.float32
assert 0 <= image.min() <= image.max() <= 1
print(f"Training: {len(train_data):,}; validation: {len(valid_data):,}; test: {len(test_data):,}")
print(f"Example: shape={tuple(image.shape)}, dtype={image.dtype}, label={class_names[label]}")
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for ax, index in zip(axes.flat, range(10)):
    image, label = train_data[index]
    ax.imshow(image.squeeze().numpy(), cmap="gray")
    ax.set_title(class_names[label], fontsize=10)
    ax.axis("off")
fig.suptitle("Fashion-MNIST training examples")
fig.tight_layout()
fig.savefig(ARTIFACTS / "training_examples.png", bbox_inches="tight")
plt.show()
plt.close(fig)
