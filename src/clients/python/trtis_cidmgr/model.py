# Copyright (c) 2019, Doug Napoleone. All rights reserved.

import sys, os, os.path
import argparse
import shutil
from . import util
from .version import __version__

_LIB= util.name2lib('cidmgr')

_template = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'config.pbtxt.in')

def dirtype(dirname):
    full_path = util.expand(dirname)
    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        raise argparse.ArgumentTypeError('Could not find directory.')
    return full_path

parser = argparse.ArgumentParser(version=__version__, add_help=True,
    description="""Install the cidmgr model into the trtserver model-store.

    This tool does not contain the custom backend shared library. 
    You can specify the --i option to search for this library on the system 
    and if found, install that. Otherwise this tool will just set up the 
    directory structure and the config.pbtxt in the TRTIS Model Repository 
    (a.k.a. model-store.)
    """)
parser.add_argument("store", metavar='model-store', type=dirtype,
                    help="trtserver model repository directory")
parser.add_argument("-o", "--overwrite", action='store_true',
                    help="overwrite to an existing model of NAME if present")
parser.add_argument("-n", "--name", nargs='?', default="cidmgr", 
                    help="model name (DEFAULT: cidmgr)")
parser.add_argument("-m", "--modver", dest='version', 
                    nargs='?', type=int, default=1, 
                    help="model version (DEFAULT: 1)")
parser.add_argument("-l", "--library", nargs='?', default=_LIB, 
                    help="model version (DEFAULT: "+_LIB+")")
parser.add_argument("-i", "--install", action='store_true', 
                    help="search for LIBRARY and install in the model")
parser.add_argument("-p", "--path", nargs="+",  default=[],
                    help="additional search paths for finding LIBRARY (implicit '-i')")
def main():
    args = parser.parse_args()
    if args.path:
        args.install=True
    modeldir = os.path.join(args.store, args.name)
    modelvdir = os.path.join(modeldir, str(args.version))

    modlib = None
    if args.install:
        modlib = util.find_library(args.library, args.path)
        if not modlib:
            parser.error('Could not find library to install: ' + args.library)
        # overwrite library name to get the exact library name found.
        # e.g.: -l foo becomes libfoo.so.0.1
        args.library = os.path.basename(modlib)

    if os.path.exists(modeldir):
        if not os.path.isdir(modeldir):
            parser.error("Supplied model directory exists "
                "but is not a directory:\n    "+ modeldir)
        elif not args.overwrite:
            parser.error("Model directory already exists:\n    "+
                modeldir+"\nUse '-o' to overwrite")
    else:
        os.mkdir(modeldir)
    if not os.path.exists(modelvdir):
        os.mkdir(modelvdir)
    _config = os.path.join(modeldir, 'config.pbtxt')
    with open(_template, 'rU') as t:
        template = t.read()
        config = template % (args.name, args.library)
        with open(_config, 'w') as c:
            c.write(config)
        print("Wrote config: "+_config)
    if modlib:
        dest = os.path.join(modelvdir, args.library)
        shutil.copyfile(modlib, dest)
        print("Wrote custom backend: " + dest)


if __name__ == '__main__':
    main()