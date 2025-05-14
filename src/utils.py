"""
This module includes utility functions related to generic data structures 
to simplify operations during training and evaluation.

Author: Sahar Jahani
"""

import logging
logger = logging.getLogger(__name__) 

        
def support_count(list):
    """
    gets a list and returns the number of elements that are greater than zero
    """
    counter = 0
    for item in list:
        if item > 0:
            counter += 1
    return counter

def set_job_name(name: str) -> None:
    """Set the global job name."""
    global job_name
    job_name = name



