"""Run the interactive demo with the model weights in this checkout."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    os.environ["USE_LOCAL_MODELS"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from app import app

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
