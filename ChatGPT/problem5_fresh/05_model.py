class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 300), nn.ReLU(),
                                 nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 10))
    def forward(self, images):
        return self.mlp(images)

torch.manual_seed(SEED)
model = ImageClassifier().to(device)
parameter_count = sum(p.numel() for p in model.parameters())
assert parameter_count == 266610
print(model)
print("Parameters:", parameter_count)
