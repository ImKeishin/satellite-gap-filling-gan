from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset

train_root = r"C:\licenta_data\SEN12MSCRTS\train\africa"
test_root = r"C:\licenta_data\SEN12MSCRTS\test\s2_africa_test"

train_ds = SEN12MSCRTSDataset(
    train_root,
    split="train",
    region="africa",
    cloud_detector="heuristic",
)

val_ds = SEN12MSCRTSDataset(
    train_root,
    split="val",
    region="africa",
    cloud_detector="heuristic",
)

test_ds = SEN12MSCRTSDataset(
    test_root,
    split="test",
    region="africa",
    cloud_detector="heuristic",
)

print("Antrenare =", len(train_ds))
print("Validare =", len(val_ds))
print("Testare =", len(test_ds))