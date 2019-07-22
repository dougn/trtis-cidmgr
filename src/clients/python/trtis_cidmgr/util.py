# Copyright (c) 2019, Doug Napoleone. All rights reserved.

import sys, os, os.path
import subprocess
import ctypes

def name2lib(name):
    """Convert a name 'foo' into the OS dependent library name::
        libfoo.so
        libfoo.dylib
        foo.dll
    """
    _prefix = "" if os.name == 'nt' else 'lib'
    _dll = "dll" if os.name == '.nt' else '.so'
    if sys.platform == 'darwin':
        _dll = '.dylib'
    return _prefix + name + _dll

def lib2name(lib):
    """Convert an OS dependent library name to the base name::
        libfoo.so.0.1 => foo
        foo.dll       => foo
    """
    if lib.startswith('lib'):
        lib = lib[4:]
    return lib.split('.',1)[0]

def expand(path):
    """Return the abspath for a given path
    """
    return os.path.abspath(os.path.expanduser(path))

def find_library(libname, paths=[]):
    """Search the system (and optional paths) for a fully
    qualified library name. Uses system configurations, 
    PATH, LD_LIBRARY_PATH and DY_LDLIBRARY_PATH::
        find_library('foo') => /usr/lib/libfoo.so.0.1
    
    Search order:
        * 'paths' tuple argument in order
        * env LD_LIBRARY_PATH in order
        * env DYLD_LIBRARY_PATH in order
        * env PATH in order
        * system paths as determined by ctypes

    .. NOTE:: On OSX, the system python will often not work
              due to env restrictions. Using virtualenv
              or similar will work around this restriction
              even if based on the system python.  
    """
    paths = [expand(p) for p in paths]
    name = lib2name(libname)
    env = os.environ.copy()
    LD=env.get('LD_LIBRARY_PATH', "")
    DY=env.get('DYLD_LIBRARY_PATH', "")
    PA=env.get('PATH', "")
    search_in = paths[:]
    if LD:
        search_in.append(LD)
    if DY:
        search_in.append(DY)
    if PA:
        search_in.append(PA)
    full_search_path = os.pathsep.join(search_in)

    search_env = 'LD_LIBRARY_PATH'
    cmd_prefix = ''
    if os.name == 'nt':
        search_env = 'PATH'
    elif sys.platform == 'darwin':
        search_env = 'DYLD_LIBRARY_PATH'
        cmd_prefix = 'export %s="%s"; ' % (search_env, full_search_path)
    env[search_env]=full_search_path
    
    ## OSX really confuses things. Only way to make it work.
    ## even plumbum fails.
    command = '%s"%s" -c "import ctypes.util; print(ctypes.util.find_library(\'%s\'))"' % (
        cmd_prefix, sys.executable, name)
    sp = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE)
    found = sp.communicate()[0].strip()
    if found == 'None':
        return None
    return found
