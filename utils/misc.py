import sys


class Unbuffered:

    def __init__(self, stream, fo):
        self.stream = stream
        self.fo = fo

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fo.write(data)

    def flush(self):
        self.stream.flush()


def print_arguments(args):
    print("Command line arguments:")
    for arg in vars(args):
        print(f'  {arg} : {getattr(args, arg)}')
    print("\n")


def invoke_with_log(fun_to_run, logfile, **kwargs):
    with open(logfile, "w") as log:
        sys.stdout = Unbuffered(sys.stdout, log)
        print(f"Logging to {logfile}")
        return fun_to_run(kwargs)
