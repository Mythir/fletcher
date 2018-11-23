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

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

// Apache Arrow
#include <arrow/api.h>

std::shared_ptr<std::vector<int32_t>> genRandomLengths(int32_t amount, uint32_t min, uint32_t mask, int32_t *total);

std::shared_ptr<std::vector<char>> genRandomValues(const std::shared_ptr<std::vector<int32_t>> &lengths,
                                                   int32_t amount);

std::shared_ptr<arrow::Array> deserializeToArrow(const int32_t* lengths, const uint8_t* values, int32_t num_strings, int32_t num_chars);