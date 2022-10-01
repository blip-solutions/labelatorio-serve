# config here

import os
import json
# if using .env

# load local .env

config_json={}
if os.path.exists("/config/config.json"):
    with open("/config/config.json", mode="r")  as cf:
        config_json=json.load(cf)

if os.path.exists("app/config/config.json"):
    with open("app/config/config.json", mode="r") as cf:
        config_json=json.load(cf)

def try_get_config(config_key, default=None, is_list=False, required=True):
    if config_key in os.environ:
        if is_list:
            return os.environ[config_key].split(";")
        else:
            return os.environ[config_key]
    elif config_key in config_json and config_json[config_key]:
        return config_json[config_key]
    elif default is not None or not required:
        return default
    else:
        raise Exception(f"Unable to retrieve config {config_key}. Try setting it as an ENV variable OR provide a config file at /config/config.json")
    

NODE_NAME=try_get_config("NODE_NAME")
LABELATORIO_API_TOKEN = try_get_config("LABELATORIO_API_TOKEN")

LABELATORIO_URL = try_get_config("LABELATORIO_URL","https://api.labelator.io")

MAX_CONNECTIONS_COUNT =try_get_config("MAX_CONNECTIONS_COUNT", 10)
MIN_CONNECTIONS_COUNT =try_get_config("MIN_CONNECTIONS_COUNT", 4)
ALLOWED_HOSTS = try_get_config("ALLOWED_HOSTS" ,["*"], is_list=True)

ROOT_PATH = try_get_config("ROOT_PATH",required=False)

SERVICE_ACCESS_TOKEN =  try_get_config("SERVICE_ACCESS_TOKEN",required=False)  #this is for internal use of managed nodes


DISABLED_AUTH = try_get_config("DISABLED_AUTH",default=False)