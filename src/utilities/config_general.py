import time
import os

time_string = time.strftime("%Y%m%d-%H%M%S")

import logging

logs_folder = os.sep.join([os.environ['PWD'], 'logs'])
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

logs_filename = f'{time_string}_log.log'
logs_file = os.sep.join([logs_folder, logs_filename])
# logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')



