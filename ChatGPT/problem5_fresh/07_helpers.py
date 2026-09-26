def evaluate(model, loader):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count, "count": count}

def train_epoch(model, loader):
    model.train()
    batch_loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_loss_sum += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return {"loss": batch_loss_sum / len(loader), "accuracy": correct / count, "count": count}
