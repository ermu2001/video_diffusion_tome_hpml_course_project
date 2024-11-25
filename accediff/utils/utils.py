from omegaconf import DictConfig
from typing import Union, Dict
import time

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


def benchmark_time_iterator(num_warmup=3, logger=None):
    def _detorator(iterable_func):
        def _wrapped_iterator(*args, **kwargs):
            print_func = logger.info if logger else print
            total_time, total_iter, total_warmup = 0, 0, 0
            print_func(f"Running {iterable_func.__name__} with {num_warmup} warmup iterations")
            start_time = time.time()
            for item in iterable_func(*args, **kwargs):
                end_time = time.time()
                yield item
                if total_warmup < num_warmup:
                    total_warmup += 1
                    print_func(f"Warmup step {total_warmup}: {end_time - start_time} seconds")
                else:
                    total_iter += 1
                    total_time += end_time - start_time
                start_time = time.time()
            assert total_iter > 0, f"No iterations were run, the original iterator has less item to iterate than num_warmup {num_warmup}"
            print_func(f"Total time: {total_time} seconds for {total_iter} iterations, throughput: {total_iter / total_time} iterations/second")

        return _wrapped_iterator
    return _detorator