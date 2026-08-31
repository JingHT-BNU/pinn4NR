"""
logutil.py —— 统一日志工具
==========================

所有子项目脚本统一接入:
    - 控制台输出:  HH:MM:SS | 消息
    - 文件输出:    paper/logs/<project>/<script>_<启动时间戳>.log (含日期与级别)

用法(每个入口脚本):
    import logging
    from logutil import setup_logging

    log = logging.getLogger("paper.<project>.<script>")  # 与 setup 的名字保持一致

    def main():
        setup_logging("<project>", "<script>")
        log.info("...")
        ...

    if __name__ == "__main__":
        try:
            main()
        except Exception:
            log.exception("运行失败")
            raise

同一 project 下所有模块 logger (paper.<project>.*) 的输出都会汇聚到
入口脚本配置的这两个 handler, 子模块只需定义自己的模块级 logger 即可。
"""

import logging
import os
from datetime import datetime, timezone, timedelta

PAPER_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(PAPER_ROOT, "logs")

# 日志时间统一用 UTC+8(北京时间)固定偏移, 与运行机器的本地时区无关
# (训练服务器为 UTC, 否则日志文件名/时间戳会差 8 小时)。
_LOG_TZ = timezone(timedelta(hours=8))


def _cn_time(secs: float):
    return datetime.fromtimestamp(secs, _LOG_TZ).timetuple()


_FMT_CONSOLE = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
_FMT_FILE = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                              "%Y-%m-%d %H:%M:%S")
_FMT_CONSOLE.converter = _cn_time
_FMT_FILE.converter = _cn_time


def setup_logging(project: str, script: str, level=logging.INFO) -> str:
    """为 paper.<project> 配置控制台+文件双输出, 返回日志文件路径。

    每次运行生成带时间戳的新日志文件, 不会覆盖历史日志。
    重复调用会先清空已有 handler, 不会产生重复输出。
    """
    log_dir = os.path.join(LOG_ROOT, project)
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{script}_{datetime.now():%Y%m%d_%H%M%S}.log")

    root = logging.getLogger(f"paper.{project}")
    root.setLevel(level)
    root.propagate = False
    root.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(_FMT_CONSOLE)
    root.addHandler(ch)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(_FMT_FILE)
    root.addHandler(fh)

    root.info(f"日志文件: {logfile}")
    return logfile
