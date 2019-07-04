# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import logging
import time
import traceback
import os
import sys



def format_message(record):
    try:
        record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message


class bcolors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GlogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',  # FATAL is alias of CRITICAL
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    COLOR_MAP = {
        logging.FATAL: bcolors.PURPLE + bcolors.BOLD,  # FATAL is alias of CRITICAL
        logging.ERROR: bcolors.RED,
        logging.WARN: bcolors.YELLOW,
        logging.INFO: bcolors.GREEN,
        logging.DEBUG: bcolors.BLUE
    }

    def __init__(self, colored=False):
        logging.Formatter.__init__(self)
        self.colored = colored

    def format(self, record):
        try:
            level = GlogFormatter.LEVEL_MAP[record.levelno]
        except KeyError:
            level = '?'
        date = time.localtime(record.created)
        date_msec = (record.created - int(record.created)) * 1e3
        #record_message = '%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s' % (
        #    level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
        #    date.tm_sec, date_usec,
        #    record.process if record.process is not None else '?????',
        #    record.filename,
        #    record.lineno,
        #    format_message(record))
        record_message = '%s[%s%02d%02d %02d:%02d:%02d.%03d %s:%s]%s %s' % (
            GlogFormatter.COLOR_MAP[record.levelno] if self.colored else '',
            level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec, date_msec,
            record.module,
            record.lineno,
            bcolors.RESET if self.colored else '',
            format_message(record))
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)

def setLevel(newlevel):
    global logger
    logger.setLevel(newlevel)
    logger.debug('Log level set to %s', newlevel)



debug = logging.debug
info = logging.info
warning = logging.warning
warn = logging.warning
error = logging.error
exception = logging.exception
fatal = logging.fatal
log = logging.log

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
WARN = logging.WARN
ERROR = logging.ERROR
FATAL = logging.FATAL

_level_names = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARN: 'WARN',
    ERROR: 'ERROR',
    FATAL: 'FATAL'
}

_level_letters = [name[0] for name in _level_names.values()]

GLOG_PREFIX_REGEX = (
    r"""
    (?x) ^
    (?P<severity>[%s])
    (?P<month>\d\d)(?P<day>\d\d)\s
    (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d)
    \.(?P<microsecond>\d{6})\s+
    (?P<process_id>-?\d+)\s
    (?P<filename>[a-zA-Z<_][\w._<>-]+):(?P<line>\d+)
    \]\s
    """) % ''.join(_level_letters)
"""Regex you can use to parse glog line prefixes."""


def setup_logger(library_name: str, logging_path: str, level: str = 'INFO') -> logging.Logger:

    directory = os.path.split(logging_path)[0]

    # If the directory does not exist, create it
    # if not os.path.exists(directory):
   #     os.makedirs(directory)

    logger = logging.getLogger(library_name)
    logger.setLevel(logging.DEBUG)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(logging_path + 'run.log')
    fh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = GlogFormatter()
    fh.setFormatter(formatter)
    # Add the handlers to the tb_logger
    logger.addHandler(fh)

    # Create console handler with a higher log level
    formatter_colored = GlogFormatter(colored=True)
    ch = logging.StreamHandler(sys.stdout)
    if level == 'INFO':
        ch.setLevel(logging.INFO)
    elif level == 'DEBUG':
        ch.setLevel(logging.DEBUG)
    else:
        return None

    ch.setFormatter(formatter_colored)
    logger.addHandler(ch)

    return logger