"""
logger.py — 日志系统初始化
==============================
统一配置项目的日志输出格式、级别、文件输出等。

使用方式:
    from utils.logger import setup_logger
    setup_logger(cfg)  # 项目启动时调用一次

    # 其他模块中直接使用标准 logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info("模块已加载")
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    cfg: dict = None,
    log_dir: str = "logs/",
    level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:
    """
    初始化全局日志系统。

    日志格式: [时间] [级别] [模块名] 消息
    输出渠道:
      - 控制台 (stdout): 彩色输出
      - 日志文件: 按日期命名，保存在 log_dir 下

    参数:
        cfg (dict):        配置字典（可选，从中读取 system.log_dir）
        log_dir (str):     日志文件输出目录
        level (str):       日志级别 ("DEBUG" / "INFO" / "WARNING" / "ERROR")
        log_to_file (bool): 是否同时输出到文件

    返回:
        root_logger: 根 Logger 实例
    """
    if cfg is not None:
        log_dir = cfg.get("system", {}).get("log_dir", log_dir)

    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 获取根 Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if root_logger.handlers:
        return root_logger

    # ---- 日志格式 ----
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # ---- 控制台 Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ---- 文件 Handler ----
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"tracking_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的 DEBUG 级别
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"日志文件: {log_file}")

    root_logger.info("日志系统初始化完成")
    return root_logger
