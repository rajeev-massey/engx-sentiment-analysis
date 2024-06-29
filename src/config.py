import yaml

class Config:
    def __init__(self, config_file):
        try:
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Please check the file path.")
            self.config = {}

    def get(self, key):
        return self.config.get(key, None)


def get(param):
    return None