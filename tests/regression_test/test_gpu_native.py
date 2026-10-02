#!/usr/bin/env python
# coding: utf-8

#
# A python script for GENESIS regression tests of the device-native GPU core
# (SPDYN built with --enable-gpu; the [DYNAMICS] gpu_resident keyword)
#
# usage: ./test_gpu_native.py "mpirun -np 4 /path/to/spdyn"
#
# A declined run falls back to the stock path, so a test passes only when its
# output has the native core's engagement line (effective=native) and no
# decline (effective=cpu).  A test directory holds
#   inp            the control file
#   ref            the reference output (energies compared as in test.py)
#   expect_abort   instead of ref: the run must stop, and its output must
#                  contain this file's text
#
# (c) Copyright 2022 RIKEN. All rights reserved.
#

import subprocess
import os
import os.path
import sys
import re
from genesis import *

############### DEFINITION ##################################
def getdirs(path):
    test_dirs = []
    for system_dir in sorted(os.listdir(path)):
        system_dir_path = os.path.join(path, system_dir)
        if not os.path.isdir(system_dir_path):
            continue
        for test_dir in sorted(os.listdir(system_dir_path)):
            test_dir_path = os.path.join(system_dir_path, test_dir)
            if os.path.exists(test_dir_path + "/inp") and \
               (os.path.exists(test_dir_path + "/ref") or
                os.path.exists(test_dir_path + "/expect_abort")):
                test_dirs.append(test_dir_path)
    return test_dirs

############### MAIN ########################################

tolerance = 1.0e-6        # relative energy difference, DOUBLE
tolerance_single = 3.0e-5 # relative energy difference, MIXED
virial_ratio = 2.0e2
pattern_native = re.compile(r'^Setup_GPU_Core> requested=YES effective=native ')
pattern_cpu = re.compile(r'^Setup_GPU_Core> requested=YES effective=cpu ')

if len(sys.argv) < 2:
    print("usage: %s \"mpirun -np N /path/to/spdyn\" [tolerance]" % sys.argv[0])
    sys.exit(3)
genesis_command = sys.argv[1]
if len(sys.argv) > 2:
    tolerance = float(sys.argv[2])

split = genesis_command.split()
split[-1] = os.path.abspath(os.path.expanduser(split[-1]))
genesis_command = " ".join(split)
if genesis_command[-5:] != "spdyn":
    print("Error: these tests are for spdyn")
    sys.exit(3)

ipassed = 0
ifailed = 0
iaborted = 0

test_dirs = getdirs(os.path.dirname(os.path.abspath(__file__)) +
                    "/test_gpu_native_spdyn")

print("=======================================================================")
print(" Regression tests for the native GPU core")
print("=======================================================================")

cwdname = os.getcwd()
for dirname in test_dirs:
    os.chdir(dirname)
    print("-----------------------------------------------------------------------")
    print("Running %s..." % (dirname + "/"))
    commandline = '%s inp 1> test 2> error' % genesis_command
    print("$ %s" % commandline)
    status = subprocess.getstatusoutput(commandline)
    output = open("test").read() + open("error").read()

    if os.path.exists("expect_abort"):
        want = open("expect_abort").read().strip()
        if status[0] != 0 and want in output:
            print("Stopped as expected (%s)" % want)
            ipassed = ipassed + 1
        else:
            print("Failed: the run was expected to stop with \"%s\"" % want)
            ifailed = ifailed + 1
        os.chdir(cwdname)
        continue

    if (status[0] > 0) and (status[0] != 1024):
        print()
        print("Aborted...")
        print()
        iaborted = iaborted + 1
        os.chdir(cwdname)
        continue
    if not any(pattern_native.match(line) for line in output.splitlines()) \
       or any(pattern_cpu.match(line) for line in output.splitlines()):
        print("Failed: the native core did not engage")
        ifailed = ifailed + 1
        os.chdir(cwdname)
        continue

    test = Genesis()
    test.read("test")
    ref = Genesis()
    ref.read("ref")
    tolerance_cur = tolerance_single if test.is_single else tolerance
    print()
    print("Checking diff between ref and test...")
    print()
    ref.test_diff(test, tolerance_cur, tolerance_cur * virial_ratio)
    print()
    if ref.is_passed:
        ipassed = ipassed + 1
    else:
        ifailed = ifailed + 1
    os.chdir(cwdname)

print()
print("%d / %d passed, %d failed, %d aborted" %
      (ipassed, len(test_dirs), ifailed, iaborted))
if ifailed > 0 or iaborted > 0:
    sys.exit(1)
sys.exit(0)
