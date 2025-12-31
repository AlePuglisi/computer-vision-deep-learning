import logging 

# Basic logging settings 
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', 
    handlers=[
        logging.FileHandler("app1.log"),
        logging.StreamHandler() #allow to put all log in this app1.log
    ]
)

logger = logging.getLogger("ArithmethicApp")

def add(a, b):
    result = a + b
    logger.debug(f"Adding {a} + {b} = {result}")
    return result 

def subtract(a, b):
    result = a - b
    logger.debug(f"Subtract {a} - {b} = {result}")
    return result 

def multiply(a, b):
    result = a * b
    logger.debug(f"AddMultiply {a} * {b} = {result}")
    return result 

def divide(a, b):
    try:
        result = a / b
        logger.debug(f"Divide {a} / {b} = {result}")
        return result 
    except ZeroDivisionError: 
        logger.error("Division by zero error")
        return None
    
add(10, 15)
subtract(15, 10)
multiply(10, 20)
divide(20, 10)
divide(1, 0)

