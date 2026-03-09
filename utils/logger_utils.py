import logging
logger = logging.getLogger("main")

import sys
import os
import datetime
from zoneinfo import ZoneInfo

from configs.global_setting import timezone


def set_logger(output_folder_dir, args):
    # Configure timezone for log timestamps
    my_timezone = ZoneInfo(timezone)

    # Define log format: timestamp | level : message
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")

    # Override default time converter to use configured timezone
    log_formatter.converter = lambda *args: datetime.datetime.now(my_timezone).timetuple()

    # File handler: write logs to exp.log in output folder (overwrite mode)
    os.makedirs(output_folder_dir, exist_ok=True)
    file_handler = logging.FileHandler(output_folder_dir + 'exp.log', mode = 'w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler: print logs to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Set minimum logging level to INFO
    logger.setLevel(logging.INFO)

    return logger