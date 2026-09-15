import logging


def silence_spam() -> None:
    logging.captureWarnings(capture=True)
    logger = logging.getLogger("py.warnings")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
    logger.addFilter(lambda record: "UserWarning: Lazy modules" not in record.getMessage())


def enable_spam() -> None:
    logging.captureWarnings(capture=False)
