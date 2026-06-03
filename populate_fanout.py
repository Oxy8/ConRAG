from __future__ import annotations

import logging

from simple_parsing import parse

from conrag.config import Config
from conrag.common import configure_logging
from conrag.fanout_dataset import build_fanout_dataset

logger = logging.getLogger("conrag.populate_fanout")


def main() -> int:
    config = parse(Config)
    configure_logging(config)
    logger.info("Starting FanOutQA dataset population")
    dataset_dir = build_fanout_dataset(config)
    logger.info("Finished FanOutQA dataset population at %s", dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
