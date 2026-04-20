
class Struct:

    def __init__(self, dictionary={}):
        for name, value in dictionary.items():
            self.__setattr__( name, value)
        
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name in {"__getstate__", "__setstate__"}:
            raise AttributeError
        return self.__dict__.get(name, None)

    def to_dict(self):
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, Struct):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        else:
            return value

    def peek(self):
        for key, value in self.__dict__.items():
            if isinstance(value, Struct):
                print(f"{key}: [Nested Struct]")
            elif isinstance(value, list):
                print(f"{key}: [List containing {len(value)} items]")
            else:
                print(f"{key}: {value}")
