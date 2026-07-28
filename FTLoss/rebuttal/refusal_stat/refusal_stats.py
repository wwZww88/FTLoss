import json
import re
import sys
import statistics

PATTERNS = [
    r"i'?m sorry",
    r"i am sorry",
    r"i apologi[sz]e",
    r"i can'?t help",
    r"i can'?t provide",
    r"i can'?t assist",
    r"i can'?t do",
    r"i cannot help",
    r"i cannot provide",
    r"i cannot assist",
    r"i'?m not able to",
    r"i am not able to",
    r"i'?m unable to",
    r"i am unable to",
    r"i must decline",
    r"i have to decline",
    r"i refuse to",
    r"it'?s not appropriate",
    r"it is not appropriate",
    r"no, i cannot",
    r"no, i can'?t",
    r"no, i will not",
    r"no, i won'?t",
]
RE = re.compile("|".join("(?:" + p + ")" for p in SPECIFIC_PATTERNS), re.IGNORECASE)


def has_refusal_in_head(text, regex, window=120):
    return bool(regex.search(text[:window]))


def has_refusal_anywhere(text, regex):
    return bool(regex.search(text))


def collect_texts(data):
    safe_texts, unsafe_texts = [], []
    for concept, v in data.items():
        safe_texts.extend(v["safe"])
        unsafe_texts.extend(v["unsafe"])
    return safe_texts, unsafe_texts


def analysis_broad(safe_texts, unsafe_texts, window=120):
    """head-of-text and anywhere."""
    print("=" * 70)
    print(f"(1) Broad refusal openers  (head window = {window} chars)")
    print("=" * 70)
    for cat, texts in [("safe", safe_texts), ("unsafe", unsafe_texts)]:
        total = len(texts)
        head = sum(has_refusal_in_head(t, BROAD_RE, window) for t in texts)
        anyw = sum(has_refusal_anywhere(t, BROAD_RE) for t in texts)
        print(
            f"  {cat:6s}: total={total}, "
            f"refusal-in-head={head} ({100*head/total:.1f}%), "
            f"refusal-anywhere={anyw} ({100*anyw/total:.1f}%)"
        )
    print()


def analysis_specific(safe_texts, unsafe_texts):
    """Analysis (2): specific model-refusal patterns (anywhere in text)."""
    print("=" * 70)
    print("(2) Specific model-refusal patterns (verb-object / apology)")
    print("=" * 70)
    for cat, texts in [("safe", safe_texts), ("unsafe", unsafe_texts)]:
        total = len(texts)
        cnt = sum(bool(SPECIFIC_RE.search(t)) for t in texts)
        print(f"  {cat:6s}: {cnt}/{total} = {100*cnt/total:.1f}%")
    print()


def analysis_length(safe_texts, unsafe_texts):
    """Auxiliary: word-count comparison as a style/format proxy."""
    print("=" * 70)
    print("(aux) Word-count comparison (style/length proxy)")
    print("=" * 70)
    sl = [len(t.split()) for t in safe_texts]
    ul = [len(t.split()) for t in unsafe_texts]
    print(f"  safe  : mean={statistics.mean(sl):.1f}, median={statistics.median(sl):.0f}")
    print(f"  unsafe: mean={statistics.mean(ul):.1f}, median={statistics.median(ul):.0f}")
    print()


def print_examples(safe_texts, unsafe_texts, regex, window=120, n=5, snippet=160):
    """Show a few flagged samples for manual inspection."""
    print("=" * 70)
    print("(examples) refusal-flagged samples (head match)")
    print("=" * 70)
    for cat, texts in [("SAFE", safe_texts), ("UNSAFE", unsafe_texts)]:
        print(f"--- {cat} ---")
        shown = 0
        for t in texts:
            if has_refusal_in_head(t, regex, window):
                print("  -", t[:snippet].replace("\n", " "))
                shown += 1
                if shown >= n:
                    break
        print()


def main():
    


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_dataset_plain.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} concepts from {path}\n")

    safe_texts, unsafe_texts = collect_texts(data)

    analysis_broad(safe_texts, unsafe_texts, window=120)
    analysis_specific(safe_texts, unsafe_texts)
    analysis_length(safe_texts, unsafe_texts)
    print_examples(safe_texts, unsafe_texts, BROAD_RE, window=120, n=5)
