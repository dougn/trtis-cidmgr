#!/usr/bin/env python

## run 50 clients in parallel with 2 connections each to trtis server
import multiprocessing
import os
import sys
import argparse

## silly script to hammer on the server in parallel and cause problems for sequence models.


def run(prog):
    if prog.endswith('.py'):
        os.system(sys.executable+" " +prog)
    else:
        os.system(prog)

def queue(prog, executions):
    pl = []
    for i in range(executions):
        p = multiprocessing.Process(target=run, args=(prog,))
        p.start()
        pl.append(p)
    for p in pl:
        p.join()

def runmany(prog, parallel=10, executions=5):
    """Run a supplied program or python script in a highly parallel way.
    parallel is the number of parallel executors to create, each of which will
    start up executions of the program each in parallel again.

    By default this will start 10 parallel executors which each are starting
    5 of the processes running in parallel for 50 total parallel runs of the
    supplied program. This is designed to generate the most race conditions
    possible.
    """
    pl = []
    for i in range(parallel):
        p = multiprocessing.Process(target=queue, args=(prog, executions,))
        p.start()
        pl.append(p)

    for p in pl:
        p.join()

parser = argparse.ArgumentParser(description=runmany.__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--parallel', type=int, default=10, 
    help="Number of executor threads to start running")
parser.add_argument('-e', '--executions', type=int, default=5, 
    help="Number of executions in parallel started by each executor thread.")
parser.add_argument('program',
    help="Program to execute in parallel. If it ends with .py, it will be run with the python interpreter running this script.")

def main():
    args = parser.parse_args()
    runmany(args.program, args.parallel, args.executions)

if __name__ == '__main__':
    main()
