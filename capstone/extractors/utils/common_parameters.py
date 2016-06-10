import json
import os


class Parameters():
    """
    Class to package the commonly used parameters by all extractors
    """
    def __init__(self, params_filename):
        with open(params_filename) as params_file:
            params = json.load(params_file)
            for param in params:
                setattr(self, param, params[param])