import argparse
import json
import re
from statistics import mean, median


def load_rows(path):
    with open(path, "r", encoding="utf-8") as fin:
        content = fin.read()

    if content.lstrip().startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def summarize(values):
    return {
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def print_summary(title, stats):
    print(title)
    print(f"  Среднее: {stats['mean']:.2f}")
    print(f"  Медиана: {stats['median']:.2f}")
    print(f"  p90: {stats['p90']:.2f}")
    print(f"  p95: {stats['p95']:.2f}")
    print(f"  p99: {stats['p99']:.2f}")
    print(f"  Минимум: {stats['min']}")
    print(f"  Максимум: {stats['max']}")


def main():
    parser = argparse.ArgumentParser(description="Статистика длины текстов в JSON или JSONL.")
    parser.add_argument("--input", help="Путь к JSON/JSONL-файлу.")
    parser.add_argument("--text-field", default="text", help="Поле с текстом, по умолчанию text.")
    args = parser.parse_args()

    rows = load_rows(args.input)
    texts = [row.get(args.text_field) for row in rows if isinstance(row, dict)]
    texts = [text for text in texts if isinstance(text, str) and text]
    skipped = len(rows) - len(texts)

    if not texts:
        raise SystemExit(f"В поле {args.text_field!r} не найдено непустых текстов.")

    character_lengths = [len(text) for text in texts]
    word_lengths = [len(re.findall(r"\S+", text)) for text in texts]

    print(f"Количество текстов: {len(texts)}")
    print(f"Пропущено записей: {skipped}")
    print_summary("\nСимволы (включая пробелы и пунктуацию):", summarize(character_lengths))
    print_summary("\nСлова:", summarize(word_lengths))


if __name__ == "__main__":
    main()
