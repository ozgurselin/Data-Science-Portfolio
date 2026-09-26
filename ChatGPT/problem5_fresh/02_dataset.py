transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
data_root = Path(os.environ.get("FASHION_MNIST_ROOT", "datasets"))
all_training = torchvision.datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
classes = all_training.classes
image, label = all_training[0]
assert image.shape == (1, 28, 28) and image.dtype == torch.float32
assert 0 <= image.min() <= image.max() <= 1
print("Image:", tuple(image.shape), image.dtype, "label:", classes[label])
