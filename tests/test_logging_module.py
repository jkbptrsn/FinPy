import logging

logger = logging.getLogger(__name__)


def do_something():
    logger.debug("module")
    logger.info("module")
    logger.warning("module")
    logger.error("module")
