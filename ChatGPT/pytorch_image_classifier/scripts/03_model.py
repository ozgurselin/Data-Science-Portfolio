"""Stage 3: define the reference multilayer perceptron."""
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs=784, n_hidden1=300, n_hidden2=100, n_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(n_inputs, n_hidden1), nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2), nn.ReLU(),
            nn.Linear(n_hidden2, n_classes))

    def forward(self, inputs):
        return self.mlp(inputs)

torch.manual_seed(SEED)
model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert model(torch.zeros(2, 1, 28, 28, device=device)).shape == (2, 10)
assert parameter_count == 266_610
print(model)
print(f"Trainable parameters: {parameter_count:,}")
