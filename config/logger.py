from loguru import logger

def ensure_name(record: dict) -> bool:
    """Ensure every log record has a 'name' in extra; default to 'root' if missing."""
    if "name" not in record["extra"]:
        record["extra"]["name"] = "root"
    return True

def setup_logger(log_level: str, log_file: str):
    """Configure the logger with the specified log level and optional log file."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "INFO"
        logger.warning("Invalid LOG_LEVEL, using default: INFO")

    # Add console logger with custom filter to ensure 'name' is present
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=log_format,
        level=log_level,
        colorize=True,
        filter=ensure_name,
    )

    logger.info(f"Logger level: {log_level}")

    if log_file:
        try:
            # Add file logger with rotation, retention, and compression
            logger.add(
                sink=log_file,
                format=log_format,
                level=log_level,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                encoding="utf-8",
                filter=ensure_name,
            )
            logger.info(f"Logger file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to add file logger: {e}")

def get_logger(name: str):
    """Return a logger instance bound to the specified name."""
    if name:
        return logger.bind(name=name)
    return logger.bind(name="root")