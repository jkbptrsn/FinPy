import datetime
import time


def execution_time(func):
    """Decorator for timing function execution."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        delta = datetime.timedelta(seconds=end - start)
        print(f"Execution time for {func.__name__}: {delta}")
        return result
    return wrapper
