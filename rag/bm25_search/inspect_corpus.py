import json, itertools
from config import CORPUS_JSONL

def main():
    with open(CORPUS_JSONL, "r") as f:
        for line in itertools.islice(f, 3):
            rec = json.loads(line)
            print(rec.keys())
            print(rec.get("title"), rec.get("year"), rec.get("venue"))
            print("---")

if __name__ == "__main__":
    main()

