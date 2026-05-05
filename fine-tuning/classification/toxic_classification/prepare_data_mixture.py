import argparse
import pandas as pd


def data_read(path: str):
    return pd.read_json(path, lines=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-1", type=str, default="data/countries_classification/train_raw.jsonl", help="Path to the first input JSONL file.")
    parser.add_argument( "--input-file-2", type=str, default="data/toxicity_classification/ru_social_tox_stance_06_04_train.jsonl", help="Path to the second input JSONL file.")
    parser.add_argument("--output-file", type=str, default="data/toxicity_classification/train_mixed_stance_tox.jsonl", help="Path to the output mixed JSONL file.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of rows to sample from the first dataset.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for sampling and shuffling.")
    parser.add_argument("--toxicity-fill-value", type=str, default="UNDEF", help="Value used to fill missing values in the toxicity_label column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_data = data_read(args.input_file_1)
    refined_train = train_data.sample(frac=args.fraction, random_state=args.random_state)

    train_data_addition = data_read(args.input_file_2)
    refined_train = pd.concat([refined_train, train_data_addition], ignore_index=True)

    refined_train = refined_train.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    if "toxicity_label" in refined_train.columns:
        refined_train = refined_train.fillna(value={"toxicity_label": args.toxicity_fill_value})

    refined_train.to_json(args.output_file, orient="records", lines=True, force_ascii=False)

    print(f"Saved {len(refined_train)} rows to {args.output_file}")


if __name__ == "__main__":
    main()