# Script 06 - Train the classifier with SGD, tracking accuracy with torchmetrics
# The book discards train2's return value (`_ = train2(...)`); we keep it as
# `history` so the training accuracy can be plotted in script 07.
n_epochs = 20
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
history = train2(model, optimizer, xentropy, accuracy, train_loader,
                 valid_loader, n_epochs)
