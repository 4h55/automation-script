import yaml
import os
class read_config:
    def __init__(self,yaml_path="config.yaml"):
        self.yaml_path=yaml_path
        self.data=self.load()
    def load(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"配置文件不存在：{self.yaml_path}")
        with open(self.yaml_path,"r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"YAML 语法错误：{e}")
    def get(self,key_path):
        keys=key_path.split(".")
        value=self.data
        try:
            for key in keys:
                if key.isdigit():
                    key = int(key)
                value = value[key]
            return value
        except KeyError as e:
            return e



