import yaml
from pathlib import Path
from box import ConfigBox
import os, json
def read_yaml_file(path: Path)-> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    
    Args:
        path (Path): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    with open(path) as file:
        try:
            data = yaml.safe_load(file)
            return ConfigBox(data)
        except yaml.YAMLError as exc:
            print(exc)
            return None  # Return None if there's an error
# @ensure_annotations
def create_directories(path_to_dirs:list, verbose=True):
    '''
    create the list of directorres

    '''
    for path in path_to_dirs:
        os.makedirs(path, exist_ok= True)
        if verbose:
            print(f'created directory at: {path}')


# @ensure_annotations
def save_json_file(path: Path, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"json file saved at: {path}")