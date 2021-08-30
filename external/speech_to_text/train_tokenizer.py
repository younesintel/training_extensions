import argparse
import os
import json
# tokenizer
from core.tokenizer import CharLevelTokenizer
# librispeech data
from core.data import Librispeech


def main(args):
    data = Librispeech(root=args.data_path, url=args.url, download=True, transforms=None)
    tokenizer = CharLevelTokenizer(CharLevelTokenizer.train(data))

    with open(args.output_file, 'w') as f:
        json.dump(tokenizer.state_dict(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--data-path", type=str, default="./", help="path to dataset")
    parser.add_argument("--url", type=str, default="dev-clean", help="type of dataset")
    parser.add_argument("--output-file", type=str, default="./configs/tokenizer.json", help="path to output tokenizer")
    args = parser.parse_args()
    main(args)
