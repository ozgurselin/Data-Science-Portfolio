"""Stage 4: train for 20 epochs, tracking sample-weighted metrics."""
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += targets.size(0)
    return loss_sum / count, correct / count

history = []
best_valid_accuracy = -1.0
best_state = None
start_time = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * targets.size(0)
        correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        count += targets.size(0)
    valid_loss, valid_accuracy = evaluate(model, valid_loader)
    row = dict(epoch=epoch, train_loss=loss_sum/count, train_accuracy=correct/count,
               valid_loss=valid_loss, valid_accuracy=valid_accuracy)
    history.append(row)
    if valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch:02d}/{EPOCHS}: loss={row['train_loss']:.4f}, "
          f"train accuracy={row['train_accuracy']:.2%}, "
          f"validation accuracy={valid_accuracy:.2%}", flush=True)
training_seconds = time.perf_counter() - start_time
assert len(history) == EPOCHS
assert all(np.isfinite(list(row.values())).all() for row in history)
assert all(0 <= row[key] <= 1 for row in history
           for key in ["train_accuracy", "valid_accuracy"])
with (ARTIFACTS / "history.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
print(f"Training time: {training_seconds:.1f} seconds; best validation epoch: {best_epoch}")
