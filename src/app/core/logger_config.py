# src/app/core/logger_config.py
import logging
import sys
import os
from datetime import datetime

# --- Logging Directory Configuration ---
# Define the directory where log files will be stored.
LOG_DIR = "logging"
# Create the log directory if it does not already exist.
# exist_ok=True prevents an error if the directory already exists.
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging():
    """
    Configures the Python logging system to output logs to both the console
    and a daily rotating file.

    Logs will be captured at the DEBUG level and above (INFO, WARNING, ERROR, CRITICAL).
    A new log file is created each day, named after the current date (YYYY-MM-DD.log).
    Existing logs for the day will be appended to the corresponding file.
    """
    # Get the current date to name the log file.
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    # Construct the full path for the daily log file.
    log_file_path = os.path.join(LOG_DIR, f"{current_date_str}.log")

    # Remove any existing handlers from the root logger.
    # This is crucial to prevent duplicate log messages if setup_logging() is called multiple times,
    # or if other modules have already configured the root logger.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the basic logging settings.
    logging.basicConfig(
        # Set the minimum logging level to DEBUG. This means all messages
        # (DEBUG, INFO, WARNING, ERROR, CRITICAL) will be processed by handlers.
        level=logging.DEBUG,
        # Define the format for log messages.
        # %(asctime)s: Timestamp of the log record.
        # %(msecs)03d: Milliseconds part of the timestamp.
        # %(levelname)-7s: Log level (e.g., INFO, DEBUG), left-aligned, 7 characters wide.
        # %(name)s: Name of the logger (usually the module name).
        # %(message)s: The actual log message.
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s - %(message)s",
        # Define the format for the timestamp.
        datefmt="%H:%M:%S",
        # Specify the handlers for the root logger.
        handlers=[
            # A StreamHandler to output log messages to the standard output (console).
            logging.StreamHandler(sys.stdout),
            # A FileHandler to write log messages to a file.
            # mode='a': Appends messages to the file if it exists; creates a new file if not.
            # encoding='utf-8': Ensures proper handling of various characters.
            logging.FileHandler(log_file_path, mode="a", encoding="utf-8"),
        ],
    )
    # The 'pass' statement is used here as the function's primary purpose is
    # to configure the logging system, and it does not need to return a value.
    pass
