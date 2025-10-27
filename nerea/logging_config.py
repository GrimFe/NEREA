import logging
import sys

def setup_logging(level: int=logging.INFO) -> logging.RootLogger:
    """
    Configure global logging for the package.
    
    Parameters
    ----------
    level: int, optional
        level to display log messages.
        Default is `logging.INFO`.
    
    Returns
    -------
    logging.RootLogger
        the logger.
    """
    root_logger = logging.getLogger()

    # Avoid adding multiple handlers if setup is called multiple times
    if root_logger.handlers:
        return root_logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    return root_logger
