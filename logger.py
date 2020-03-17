import logging
import time

logger = logging.getLogger(__name__)
formatter = logging.Formatter("[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s")

now = time.gmtime(time.time())
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('log/{}_{}_{}_{}h_{}m.log'.format(now.tm_year, now.tm_mon, now.tm_mday,
                                                                     now.tm_hour, now.tm_min))

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
