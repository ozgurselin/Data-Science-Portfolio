# Step 2 - Load Fashion MNIST with TorchVision and split train/validation
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

class_names = train_and_valid_data.classes
print(f"train: {len(train_data):,} | valid: {len(valid_data):,} | "
      f"test: {len(test_data):,}")
print("classes:", class_names)
