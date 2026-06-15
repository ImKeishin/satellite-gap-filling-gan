from pathlib import Path

from src.dataset.sen12mscrts_loader import REGION_ROIS, SPLITS

TRAIN_ROOT = Path(r"C:\licenta_data\SEN12MSCRTS\train\africa")
TEST_ROOT = Path(r"C:\licenta_data\SEN12MSCRTS\test\s2_africa_test")
TIME_COUNT = 30
REGION = "africa"


def selected_rois(split: str) -> list[str]:
    allowed = set(REGION_ROIS[REGION])
    return [roi for roi in SPLITS[split] if roi in allowed]


def count_split(root: Path, split: str) -> int:
    print(f"\n=== {split.upper()} ===", flush=True)
    print(f"Root: {root}", flush=True)

    if not root.exists():
        raise FileNotFoundError(f"Folder inexistent: {root}")

    total = 0
    rois = selected_rois(split)

    for roi in rois:
        group, roi_id = roi.split("/")
        s2_root = root / group / roi_id / "S2"

        if not s2_root.exists():
            print(f"[LIPSEȘTE] {roi}: {s2_root}", flush=True)
            continue

        counts = []
        valid = True

        for time_index in range(TIME_COUNT):
            folder = s2_root / str(time_index)

            if not folder.exists():
                print(f"[INCOMPLET] {roi}: lipsește directorul temporal {time_index}", flush=True)
                valid = False
                break

            tif_count = sum(1 for _ in folder.glob("*.tif"))

            if tif_count == 0:
                print(f"[INCOMPLET] {roi}: nu există fișiere .tif în {folder}", flush=True)
                valid = False
                break

            counts.append(tif_count)

        if not valid:
            continue

        patch_count = min(counts)
        total += patch_count

        print(
            f"[OK] {roi}: {patch_count} patch-uri "
            f"(min={min(counts)}, max={max(counts)} fișiere temporale)",
            flush=True,
        )

    print(f"TOTAL {split.upper()} = {total}", flush=True)
    return total


def main() -> None:
    train_count = count_split(TRAIN_ROOT, "train")
    val_count = count_split(TRAIN_ROOT, "val")
    test_count = count_split(TEST_ROOT, "test")

    print("\n=== VALORI PENTRU TABELUL 2.2 ===", flush=True)
    print(f"Antrenare: {train_count}", flush=True)
    print(f"Validare:  {val_count}", flush=True)
    print(f"Testare:   {test_count}", flush=True)


if __name__ == "__main__":
    main()
