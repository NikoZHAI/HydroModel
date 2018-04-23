# Exceptions
class DivergentError(Exception):
    """ DivergentError """
    '''
    Error occurs when result numbers are out of machine precision and cause
    divergent floating point number calculations
    '''
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
 
class LoadDataError(Exception):
    """ Data file loading error
    
    Error occurs when result numbers are out of machine precision and cause
    divergent floating point number calculations
    """
    
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
 