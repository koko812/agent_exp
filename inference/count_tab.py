from collections import Counter

def count_tabs(path: str, max_lines: int = 20):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            n_tabs = line.count("\t")
            counts[n_tabs] += 1
            if i <= max_lines:
                print(f"{i:4d}: {n_tabs} tabs → {line.strip()[:80]}...")
    print("\nSummary (tab count → line count):")
    for k, v in sorted(counts.items()):
        print(f"{k} tabs: {v} lines")

# 例: 日本語版 MGSM
count_tabs("datasets/mgsm/mgsm_ja.tsv", max_lines=20)
