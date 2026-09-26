torch.manual_seed(SEED)
train_data, valid_data = random_split(all_training, [55_000, 5_000])
assert len(test_data) == 10_000
assert set(train_data.indices).isdisjoint(valid_data.indices)
assert len(set(train_data.indices) | set(valid_data.indices)) == 60_000
print({"training": len(train_data), "validation": len(valid_data), "test": len(test_data)})
