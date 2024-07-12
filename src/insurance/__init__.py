import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("InsuranceLogger")


def error_message_details(error: Exception, error_detail:sys):
    """
    Returns the error message and error detail.

    Args:
        error (str): The error message.
        error_detail (sys): The error detail.

    Returns:
        str: A formated string containing the error filename, line number, and message.
    """

    try:
        _, _, exc_tb = error_detail.exc_info()

        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno
        error_message = str(error)
        return f"Error occured in Python Script: name [{file_name}], Line Number: [{line_number}], Error Message: [{error_message}]"
    except (AttributeError, NameError):
        return f"Error: Unable to retrieve detailed error information: {str(error)}"
    

class CustomException(Exception):
    """
    Custom exception class.

    Args: 
        error_message (str): The error message
        error_details (sys): The error details

    Returns:
        str: A formated string containing the error filename, line number, and message.
        
    Usage:
        ShipmentException(e, sys)
    """

    def __init__(self, error_message: str, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)
        logging.info(self.error_message)
    
    def __str__(self):
        return self.error_message