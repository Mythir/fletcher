# Copyright 2018 Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import timeit
import pyarrow as pa
import numpy as np
import copy
import sys
import argparse
import pandas as pd

import pyfletcher as pf
import stringwrite


class Timer:
    def __init__(self, gc_disable=True):
        self.starttime = 0
        self.stoptime = 0
        self.gc_disable = gc_disable

    def start(self):
        if self.gc_disable:
            gc.disable()
        self.starttime = timeit.default_timer()

    def stop(self):
        self.stoptime = timeit.default_timer()
        gc.enable()

    def seconds(self):
        return self.stoptime - self.starttime


if __name__ == "__main__":
    t = Timer(gc_disable=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_exp", dest="ne", default=4,
                        help="Number of experiments to perform")
    parser.add_argument("--platform_type", dest="platform_type", default="echo", choices=["echo", "aws"],
                        help="Type of FPGA platform")
    parser.add_argument("--num_strings", dest="num_strings", default=6000,
                        help="Number of strings to deserialize")
    parser.add_argument("--min_len", dest="min_len", default=0,
                        help="Minimum string length")
    parser.add_argument("--len_msk", dest="len_msk", default=255,
                        help="Length mask")
    args = parser.parse_args()

    # Parsed args
    ne = int(args.ne)
    num_strings = int(args.num_strings)
    min_len = int(args.min_len)
    len_msk = int(args.len_msk)
    platform_type = args.platform_type

    # Timers
    t = Timer(gc_disable=False)
    t_pa = []
    t_pd = []
    t_py = []

    # Results
    r_pa = []
    r_pd = []
    r_py = []

    (lengths, values) = stringwrite.get_random_lengths_and_values(num_strings, min_len, len_msk)

    assert(sum(lengths) == len(values))

    print("Average length: " + str(sum(lengths)/len(lengths)))

    for i in range(ne):
        print("Starting experiment " + str(i))
        t.start()
        r_pa.append(stringwrite.deserialize_to_arrow(lengths, values))
        t.stop()
        t_pa.append(t.seconds())

        t.start()
        r_py.append(stringwrite.deserialize_to_list(lengths, values))
        t.stop()
        t_py.append(t.seconds())

        t.start()
        r_pd.append(stringwrite.deserialize_to_pandas(lengths, values))
        t.stop()
        t_pd.append(t.seconds())

        print("Total execution times for " + str(i+1) + " runs:")
        print("Pandas: " + str(sum(t_pd)))
        print("Native: " + str(sum(t_py)))
        print("Arrow: " + str(sum(t_pa)))
        print()
        print("Average execution times:")
        print("Pandas: " + str(sum(t_pd) / (i + 1)))
        print("Native: " + str(sum(t_py) / (i + 1)))
        print("Arrow: " + str(sum(t_pa) / (i + 1)))
        print()

    with open("Output.txt", "w") as textfile:
        textfile.write("\nTotal execution times for " + str(ne) + " runs:")
        textfile.write("\nPandas: " + str(sum(t_pd)))
        textfile.write("\nNative: " + str(sum(t_py)))
        textfile.write("\nArrow: " + str(sum(t_pa)))
        textfile.write("\n")
        textfile.write("\nAverage execution times:")
        textfile.write("\nPandas: " + str(sum(t_pd) / (i + 1)))
        textfile.write("\nNative: " + str(sum(t_py) / (i + 1)))
        textfile.write("\nArrow: " + str(sum(t_pa) / (i + 1)))

    # Find total size in bytes of Arrow array
    batch_size = 0
    for buffer in r_pa[0].buffers():
        if buffer is not None:
            batch_size += buffer.size

    print("Total size of Arrow Array: {bytes} bytes.".format(bytes=batch_size))

    pass_counter = 0
    cross_exp_pass_counter = 0

    for i in range(ne):
        if r_pa[i].equals(pa.array([x.decode("utf-8") for x in r_pd[i]])) \
                and r_pa[i].equals(pa.array([x.decode("utf-8") for x in r_py[i]])):
            pass_counter += 1

        if r_pa[0].equals(r_pa[i]):
            cross_exp_pass_counter += 1

    if pass_counter == ne and cross_exp_pass_counter == ne:
        print("PASS")
    else:
        print("ERROR ({error_counter} errors)".format(error_counter=ne - pass_counter))