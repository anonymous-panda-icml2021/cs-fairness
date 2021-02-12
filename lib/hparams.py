# Code credit: https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_multiscale_dataset.py

class HParams:
    """Dictionary of hyperparameters."""
    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)

    def update_config(self, in_string):
        """Update the dictionary with a comma separated list."""
        if not in_string:
            return self
        pairs = in_string.split(",")
        pairs = [pair.split("=") for pair in pairs]
        for key, val in pairs:
            self.dict_[key] = type(self.dict_[key])(val)
        self.__dict__.update(self.dict_)
        return self

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)

    def __repr__(self):
        return self.dict_.__repr__()
