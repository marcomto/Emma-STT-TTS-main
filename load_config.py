import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", choices=["it", "en"], default="it")
args = parser.parse_args()

with open(
    f"config/config_{args.lang}.json",
    "r",
    encoding="utf-8"
) as f:
    cfg = json.load(f)
