## run 50 clients in parallel with 2 connections each to trtis server
import multiprocessing
import os

## silly script to hammer on the server in parallel and cause problems for sequence models.

SCRIPT = "./doug_simple_sequence.py"
#SCRIPT = "./simple_sequence_client.py"

def run():
    os.system("python "+SCRIPT)

def queue():
    pl = []
    for i in range(5):
        p = multiprocessing.Process(target=run)
        p.start()
        pl.append(p)
    for p in pl:
        p.join()

pl = []
for i in range(10):
    p = multiprocessing.Process(target=queue)
    p.start()
    pl.append(p)

for p in pl:
    p.join()
