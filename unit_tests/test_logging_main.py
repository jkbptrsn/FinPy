import logging

import test_logging_module

logging.basicConfig(filename='example.log',
                    encoding='utf-8',
                    level=logging.INFO)

logging.debug("main")
logging.info("main")
logging.warning("main")
logging.error("main")

test_logging_module.do_something()
