import logging
import math
import time


def log_setup() -> logging.Logger:
    log = logging.getLogger('torchs2s')
    log.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('torchs2s.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    log.addHandler(c_handler)
    log.addHandler(f_handler)

    return log


def as_minutes(seconds: float) -> str:
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)


def time_since(since: float, percent: float) -> str:
    now = time.time()
    elapsed_seconds = now - since
    total_seconds = elapsed_seconds / percent
    remaining_seconds = total_seconds - elapsed_seconds
    return '%s (- %s)' % (as_minutes(elapsed_seconds), as_minutes(remaining_seconds))
