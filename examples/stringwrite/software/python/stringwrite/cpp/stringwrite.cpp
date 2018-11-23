// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <omp.h>

// Apache Arrow
#include <arrow/api.h>

#include "stringwrite.h"
#include "randomizer.h"

std::shared_ptr<std::vector<int32_t>> genRandomLengths(int32_t amount, uint32_t min, uint32_t mask, int32_t *total) {
  LFSRRandomizer lfsr;
  auto lengths = std::make_shared<std::vector<int32_t>>();
  lengths->reserve(static_cast<unsigned long>(amount));

  int total_length = 0;

  for (int32_t i = 0; i < amount; i++) {
    int len = min + (lfsr.next() & mask);
    total_length += len;
    lengths->push_back(len);
  }

  *total = total_length;

  return lengths;
}

std::shared_ptr<std::vector<char>> genRandomValues(const std::shared_ptr<std::vector<int32_t>> &lengths,
                                                   int32_t amount) {
  std::array<LFSRRandomizer, 64> lfsrs;

  // initialize the lfsrs as in hardware
  for (int i = 0; i < 64; i++) {
    lfsrs[i].lfsr = (uint8_t) i;
  }

  // reserve all characters in a vector
  auto values = std::make_shared<std::vector<char>>();
  values->reserve(static_cast<unsigned long>(amount));

  // For every string length
  for (const auto &len : *lengths) {
    uint32_t strpos = 0;
    // First generate a new "line" of random characters, even if it's zero length
    do {
      uint32_t bufpos = 0;
      char buffer[64] = {0};
      for (int c = 0; c < 64; c++) {
        auto val = lfsrs[c].next() & (uint8_t) 127;
        if (val < 32) val = '.';
        if (val == 127) val = '.';
        buffer[c] = val;
      }
      // Fill the cacheline
      for (bufpos = 0; bufpos < 64 && strpos < (uint32_t) len; bufpos++) {
        values->push_back(buffer[bufpos]);
        strpos++;
      }
    } while (strpos < (uint32_t) len);
  }

  return values;
}

std::shared_ptr<arrow::Array> deserializeToArrow(const int32_t* lengths, const uint8_t* values, int32_t num_strings, int32_t num_chars) {

  // Allocate space for values buffer
  std::shared_ptr<arrow::Buffer> val_buffer;
  if (!arrow::AllocateBuffer(num_chars, &val_buffer).ok()) {
    throw std::runtime_error("Could not allocate values buffer.");
  }

  // Copy the values buffer
  memcpy(val_buffer->mutable_data(), values, num_chars);

  // Allocate space for offsets buffer
  std::shared_ptr<arrow::Buffer> off_buffer;
  if (!arrow::AllocateBuffer((num_strings + 1) * sizeof(int32_t), &off_buffer).ok()) {
    throw std::runtime_error("Could not allocate offsets buffer.");
  }

  // Lengths need to be converted into offsets

  // Get the raw mutable buffer
  auto raw_ints = (int32_t *) off_buffer->mutable_data();
  int32_t offset = 0;

  for (size_t i = 0; i < num_strings; i++) {
    raw_ints[i] = offset;
    offset += lengths[i];
  }

  // Fill in last offset
  raw_ints[num_strings] = offset;

  return std::static_pointer_cast<arrow::Array>(std::make_shared<arrow::StringArray>(num_strings, off_buffer, val_buffer));
}