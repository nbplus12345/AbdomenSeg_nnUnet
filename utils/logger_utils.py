import logging
import os
import sys


# === 日志系统配置 ===
class Logger:
    def __init__(self, logger_name="Trainer", log_file="../logs/training.log"):
        # 1. 获取 logger 实例
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # 防止重复添加 Handler
        if not self.logger.handlers:
            # 2. 配置文件处理器
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)

            # 3. 配置屏幕处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_formatter)

            # 4. 挂载处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

    def info(self, message):
        """输出普通信息。"""
        self.logger.info(f"[INFO] {message}")

    def warning(self, message):
        """输出警告信息。"""
        self.logger.warning(f"[WARNING] {message}")

    def error(self, message):
        """输出错误信息。"""
        self.logger.error(f"[ERROR] {message}")

    def success(self, message):
        """输出成功信息。"""
        self.logger.info(f"[SUCCESS] {message}")
