import logging

from kata.core.config import get_config

# Library-level NullHandler so consumers get no output unless they configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
