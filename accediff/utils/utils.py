from omegaconf import DictConfig
from typing import Union, Dict

def recursive_dictconfig_to_dict(config: DictConfig, prefix: str = '') -> Dict[str, Union[str, int, float]]:
    result = {}
    
    for key, value in config.items():
        # Generate the current key path
        current_key = f"{prefix}--{key}" if prefix else key
        
        if isinstance(value, DictConfig):
            # If the value is another DictConfig, recurse
            result.update(recursive_dictconfig_to_dict(value, current_key))
        else:
            # Otherwise, it's a base value (str, int, float)
            result[current_key] = value
            
    return result
