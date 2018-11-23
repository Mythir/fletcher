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
    parser.add_argument("--num_exp", dest="ne", default=1,
                        help="Number of experiments to perform")
    parser.add_argument("--platform_type", dest="platform_type", default="echo", choices=["echo", "aws"],
                        help="Type of FPGA platform")
    parser.add_argument("--num_strings", dest="num_strings", default=20,
                        help="Number of points in coordinate system")
    parser.add_argument("--min_len", dest="min_len", default=0,
                        help="Minimum string length")
    parser.add_argument("--len_msk", dest="len_msk", default=255,
                        help="Max number of k-means iterations")
    args = parser.parse_args()

    # Parsed args
    ne = int(args.ne)
    num_strings = int(args.num_strings)
    min_len = int(args.min_len)
    len_msk = int(args.len_msk)
    platform_type = args.platform_type

    (lengths, values) = stringwrite.get_random_lengths_and_values(num_strings, min_len, len_msk)

    print(lengths.tobytes())
    print(values.tobytes())

    print(sum(lengths))
    print(len(values))

    testarray = stringwrite.deserialize_to_arrow(lengths, values)

    print(testarray)
    print(testarray.buffers()[1].size)
    print(testarray.buffers()[2].size)

    testlist = stringwrite.deserialize_to_list(lengths, values)

    print(testlist)

    testseries = stringwrite.deserialize_to_pandas(lengths, values)
