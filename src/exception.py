import sys
import logging

try:
    from . import logger as _logger
except ImportError:
    import logger as _logger

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    if exc_tb is None:
        ## Handle case when there's no active exception traceback
        error_message = "error occurred: {0}".format(str(error))
    else:
        file_name = exc_tb.tb_frame.f_code.co_filename ## this code return the file name pf error
        error_message = "error occurred in python script name[{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_details)
        
    def __str__(self):
        return self.error_message
    
