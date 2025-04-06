from loguru import logger
import os
import sys
from datetime import datetime

def get_logger(script_file=__file__):
    """
    Configures and returns a logger using loguru.
    
    Logs are sent to:
      - Console: with DEBUG level and above.
      - A rotating file: placed in <usecase1_dir>/logs/<script_name>.log
          * Rotates when the file exceeds ~2 MB.
          * Retains up to 2 rotated log files.
          * Compressed as zip.
    
    Args:
        script_file: The current script file (defaults to __file__).
        
    Returns:
        A configured loguru logger.
    """
    # Derive the script base name (e.g., "fine_tune_sbert")
    script_name = os.path.splitext(os.path.basename(script_file))[0]

    # Construct logs folder: .../usecase1/logs
    this_dir = os.path.dirname(os.path.abspath(__file__))
    usecase1_dir = os.path.dirname(this_dir)
    logs_dir = os.path.join(usecase1_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Remove default loguru sink(s)
    logger.remove()
    
    # Add a console sink (DEBUG+)
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"
    )
    
    # Define the file log path
    log_file_path = os.path.join(logs_dir, f"{script_name}.log")
    
    # Add a file sink (INFO+) with rotation (~2MB), retention (2 files), and compression (zip)
    logger.add(
        log_file_path,
        level="INFO",
        rotation="2 MB",
        retention=2,
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"
    )
    
    return logger

# Example usage:
if __name__ == "__main__":
    log = get_logger()
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
