import os
from datetime import datetime
from loguru import logger

class LoggerManager:
    """Singleton LoggerManager for Loguru logging"""
    
    LOG_DIR = "logs"
    
    @staticmethod
    def configure_logger():
        os.makedirs(LoggerManager.LOG_DIR, exist_ok=True)

        current_time = datetime.now().strftime("%Y-%m-%d")
        log_file_name = f"flexpg-gen-bi_{current_time}.log"
        log_file_path = os.path.join(LoggerManager.LOG_DIR, log_file_name)

        # Remove default handlers (if any) to prevent duplicate logs
        logger.remove()

        # Configure Loguru logging format
        logger.add(
            log_file_path,
            format="{time: MMMM D, YYYY - HH:mm:ss} {level} --- <level>{message}</level>",
        )

        return logger