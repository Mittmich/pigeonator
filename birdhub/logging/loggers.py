import logging

class EventLogger(logging.Logger):
    """Implements structured logs for birdhub"""

    LOG_FORMAT = "%(asctime)s\t%(levelname)s\t%(message)s"
    MESSAGE_FORMAT = "{event_type}\t{event_information}"

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
    
    def log_event(self, event_type, event_information):
        self.info(self.MESSAGE_FORMAT.format(event_type=event_type, event_information=event_information))