'''
日志的初始化
'''


from log import logger
import os
import sys

def init_loguru(level: str = "ERROR", rotation: str = "10 MB",retention: int = 5):
    logger.remove()
    log_file="crawler.log"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    log_format = (
            "<green>{time:MM-DD HH:mm}</green>|"
            "<level>{level}</level>|"
            "<blue>{module}:{line}</blue> |\n"
            "  <level>{message}</level>"
    )
    logger.add(
        log_path,
        format=log_format,
        level="INFO",
        retention=retention,
        rotation=rotation,
        enqueue=False,
        #compression="zip",
        encoding="utf-8",
    )
    logger.add(
        sys.stdout,
        format=log_format,
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

