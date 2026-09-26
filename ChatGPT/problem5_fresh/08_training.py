history = []
started = time.perf_counter()
for epoch in range(1, n_epochs + 1):
    training = train_epoch(model, train_loader)
    validation = evaluate(model, valid_loader)
    assert training["count"] == 55_000 and validation["count"] == 5_000
    row = {"epoch": epoch, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
           "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]}
    history.append(row)
    print(f"Epoch {epoch:02d}/{n_epochs}: train loss={row['train_loss']:.4f}, train accuracy={row['train_accuracy']:.4%}, validation accuracy={row['validation_accuracy']:.4%}", flush=True)
training_seconds = time.perf_counter() - started
assert len(history) == 20
(ARTIFACTS / "history.json").write_text(json.dumps(history, indent=2))
print(f"Full training time: {training_seconds:.2f} seconds")
