import logging


def get_logger(filename='log'):
    # Create and configure logger
    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    return logger
