import logging
import sys
import time
import functools
import inspect
from typing import Any, Callable

# Configure standard formatting for logging to stdout
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger for the application flow
logger = logging.getLogger("genai_baseline")

def get_logger(name: str):
    return logging.getLogger(f"genai_baseline.{name}")

def _abstract_repr(val: Any) -> str:
    """Creates a clean, abstract representation of function arguments and return values."""
    if val is None:
        return "None"
    
    t = type(val)
    if t in (int, float, bool):
        return str(val)
    
    if isinstance(val, str):
        if len(val) <= 60:
            return f"'{val}'"
        return f"str(len={len(val)})"
        
    if isinstance(val, list):
        return f"list(len={len(val)})"
        
    if isinstance(val, dict):
        keys = list(val.keys())
        if len(keys) <= 3:
            return f"dict(keys={keys})"
        return f"dict(len={len(val)}, keys={keys[:3]}...)"
        
    if isinstance(val, tuple):
        return f"tuple(len={len(val)})"
        
    # Check if the class name or basic representation is clean
    class_name = t.__name__
    try:
        r = repr(val)
        if len(r) <= 60 and not r.startswith("<"):
            return r
    except Exception:
        pass
        
    return f"<{class_name}>"

def log_function_call(func: Callable):
    """
    Decorator to log function entry, exit, inputs, outputs, and execution time.
    Supports both synchronous and asynchronous functions.
    """
    func_logger = get_logger(func.__module__)
    func_name = func.__qualname__

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        args_repr = [_abstract_repr(a) for a in args]
        kwargs_repr = {k: _abstract_repr(v) for k, v in kwargs.items()}
        
        func_logger.info(f"==> Call: {func_name} | Args: {args_repr} | Kwargs: {kwargs_repr}")
        
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            func_logger.info(f"<== Return: {func_name} | Success | Time: {duration:.4f}s | Result: {_abstract_repr(result)}")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            func_logger.error(f"<== Raise: {func_name} | Error: {type(e).__name__}: {str(e)} | Time: {duration:.4f}s")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        args_repr = [_abstract_repr(a) for a in args]
        kwargs_repr = {k: _abstract_repr(v) for k, v in kwargs.items()}
        
        func_logger.info(f"==> Call: {func_name} | Args: {args_repr} | Kwargs: {kwargs_repr}")
        
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            func_logger.info(f"<== Return: {func_name} | Success | Time: {duration:.4f}s | Result: {_abstract_repr(result)}")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            func_logger.error(f"<== Raise: {func_name} | Error: {type(e).__name__}: {str(e)} | Time: {duration:.4f}s")
            raise

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

