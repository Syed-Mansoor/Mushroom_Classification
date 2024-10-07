import os
import sys

def error_message_detail(error,error_detail:sys):
    """
    Function to get detailed message for the exception.
    :param error: class instance of Exception Data type with requires error message
    :param error_detail: object of sys module
    :return: error message
    """
    _,_,_exc_tb  = error_detail.exc_info()
    file_name = _exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,_exc_tb.tb_lineno,str(error)
    )

    return error_message


class MushroomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        """
        :param error_message: error message in string format
        :param error_detail: object of sys module
        """
        
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message