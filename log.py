import os
import sys

class Logger(object):
    def __init__(self, path, name, mode=None):
        log_file = os.path.join(path, name)
        print('saving log to ', path)
        self.terminal = sys.stdout
        self.file = None
        self.open(log_file,mode)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()
    def write_args(self, args):
        message = ''
        for arg in vars(args):
            message += arg + '=' + str(getattr(args, arg)) + '\t'

    def close(self):
        self.file.close()