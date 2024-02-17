#pragma once

#include "tic_toc.hpp"
namespace llog {
TicToc::Ptr CreateTimer(const std::string &name);
std::string GetAllTimingStatistics(int print_level = 100);
void PrintAllTimingStatistics(int print_level = 100);
void Reset();
void PrintLog(int print_level = 100);
void SaveLog(const std::string &_save_path);

} // namespace llog