from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "roneneldan/TinyStories"
SUBSET_SIZE = 100_000
OUTPUT_PATH = Path("data/raw/tinystories.txt")

dataset = load_dataset(DATASET_NAME)

train_split = dataset["train"]

subset = train_split.select(range(SUBSET_SIZE))

# Export to data/

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"💾 Writing to {OUTPUT_PATH}...")
written_count = 0
total_bytes = 0

with open(OUTPUT_PATH, "w", encoding="utf-8", errors="replace") as f:
    for idx, example in enumerate(subset):
        story = example["text"]
        f.write(story.strip())
        f.write("\n")

        written_count += 1
        total_bytes += len(story.encode("utf-8"))

        if (idx + 1) % 10_000 == 0:
            print(f"   ... wrote {idx + 1:,} stories")

    assert OUTPUT_PATH.exists(), f"Failed to create {OUTPUT_PATH}"
    assert (
        written_count == SUBSET_SIZE
    ), f"Expected {SUBSET_SIZE} stories, wrote {written_count}"

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Success!")
    print(f"   - File: {OUTPUT_PATH}")
    print(f"   - Stories: {written_count:,}")
    print(f"   - Size: {file_size_mb:.1f} MB")
    print(f"   - Avg story: {total_bytes // written_count} bytes")
