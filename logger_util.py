# logger_util.py

import logging

# Set up the logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info("This is a test log message.")