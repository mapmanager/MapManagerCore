class mmCoreException(Exception):
    pass

class MetadataError(Exception):
    def __init__(self, message, errors = None):            
        super().__init__(message)
            
        # Now for your custom code...
        self.errors = errors