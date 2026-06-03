from __future__ import annotations

import logging

from simple_parsing import parse

from conrag import ConRAG, Config
from conrag.common import configure_logging

logger = logging.getLogger("conrag.main")
VALID_MODES = {"run", "build", "query"}


def main() -> int:
    config = parse(Config)
    configure_logging(config)
    mode = normalize_mode(config.mode)
    logger.info("Starting ConRAG (mode=%s)", mode)

    app = ConRAG(config)
    if mode == "run":
        app.run()
    elif mode == "build":
        app.build_knowledge_base()
    else:
        app.query()

    logger.info("Finished ConRAG")
    return 0


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of: {', '.join(sorted(VALID_MODES))}")
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
