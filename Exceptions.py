class invalid_age(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
class invalid_name(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

class invalid_mane_color(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

class invalid_gender(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

