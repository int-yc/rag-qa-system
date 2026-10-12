
import time
from functools import wraps
from datetime import datetime

class Logger:

    def __init__(self, name: str = "RAG"):
        self.name = name

    def _print_log(self, level: str, msg: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] [{self.name}] [{level}] {msg}")

    def info(self, msg: str):
        self._print_log("INFO", msg)

    def error(self, msg: str):
        self._print_log("ERROR", msg)

def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ 函数 [{func.__name__}] 耗时：{end - start:.4f}秒")
        return result
    return wrapper

def retry(max_times: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for i in range(max_times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"⚠️ 第{i+1}次调用失败：{e}，{delay}秒后重试...")
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

logger = Logger()
