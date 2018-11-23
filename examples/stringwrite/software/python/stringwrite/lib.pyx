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

from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc, free
from libc.stdint cimport *
from libc.string cimport memcpy
from libcpp.string cimport string as cpp_string
from libcpp.vector cimport vector
from libcpp cimport bool as cpp_bool

import numpy as np
cimport numpy as np
import pandas as pd
from pyarrow.lib cimport *

cdef extern from "cpp/stringwrite.h" nogil:
    shared_ptr[vector[int32_t]] genRandomLengths(int32_t amount, uint32_t min, uint32_t mask, int32_t *total)
    shared_ptr[vector[char]] genRandomValues(const shared_ptr[vector[int32_t]] &lengths, int32_t amount)
    shared_ptr[CArray] deserializeToArrow(const int32_t* lengths, const uint8_t* values, int32_t num_strings, int32_t num_chars)


cpdef get_random_lengths_and_values(amount, min_len, len_msk):
    cdef int32_t total_chars
    cdef shared_ptr[vector[int32_t]] lengths = genRandomLengths(amount, min_len, len_msk, &total_chars)

    np_lengths = np.zeros((lengths.get().size(),), dtype=np.int32)
    cdef const int32_t[:] lengths_view = np_lengths
    cdef const int32_t *lengths_numpy_pointer = &lengths_view[0]
    cdef int32_t* lengths_vector_pointer = lengths.get().data()

    memcpy(<void*> lengths_numpy_pointer, <const void*> lengths_vector_pointer, lengths.get().size() * sizeof(int32_t))

    cdef shared_ptr[vector[char]] values = genRandomValues(lengths, total_chars)

    np_values = np.zeros((total_chars,), dtype=np.uint8)
    cdef const uint8_t[:] values_view = np_values
    cdef const uint8_t* values_numpy_pointer = &values_view[0]
    cdef char* values_vector_pointer = values.get().data()

    memcpy(<void*> values_numpy_pointer, <const void*> values_vector_pointer, total_chars)

    return np_lengths, np_values

cpdef deserialize_to_arrow(np_lengths, np_values):
    cdef const int32_t[:] lengths_view = np_lengths
    cdef const uint8_t[:] values_view = np_values
    cdef const int32_t *lengths = &lengths_view[0]
    cdef const uint8_t* values = &values_view[0]

    return pyarrow_wrap_array(deserializeToArrow(lengths, values, np_lengths.size, np_values.size))


cpdef deserialize_to_list(np_lengths, np_values):
    cdef const int32_t[:] lengths_view = np_lengths
    cdef const unsigned char[:] values_view = np_values
    cdef const int32_t* lengths = &lengths_view[0]
    cdef const unsigned char* values = &values_view[0]

    cdef list result = [None]*np_lengths.size

    cdef int i
    cdef int end
    cdef int begin = 0

    for i in range(np_lengths.size):
        end = begin + lengths[i]
        result[i]=values[begin:end]
        begin = end

    return result

cpdef deserialize_to_pandas(np_lengths, np_values):
    cdef const int32_t[:] lengths_view = np_lengths
    cdef const unsigned char[:] values_view = np_values
    cdef const int32_t* lengths = &lengths_view[0]
    cdef const unsigned char* values = &values_view[0]

    result = pd.Series([None] * np_lengths.size, dtype=str)

    cdef int i
    cdef int end
    cdef int begin = 0

    for i in range(np_lengths.size):
        end = begin + lengths[i]
        result[i] = values[begin:end]
        begin = end

    return result