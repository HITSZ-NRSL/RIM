#include "llog.h"
#include <fstream>
#include <vector>

namespace llog {

std::vector<std::pair<std::string, TicToc::Ptr>> timers;

TicToc::Ptr CreateTimer(const std::string &name) {
  TicToc::Ptr timer = std::make_shared<TicToc>();
  timers.emplace_back(name, timer);
  return timer;
}

std::string GetAllTimingStatistics(int print_level) {
  std::string str;
  str += "Timing Statistics [total time/count = avg. time, this time (ms)]:\n";
  for (const auto &it : timers) {
    if (it.first.find_first_not_of(' ') < print_level) {
      str += it.first + ":\t" + std::to_string(it.second->sum() * 1000.0) +
             "/" + std::to_string(it.second->times()) + " = " +
             std::to_string(it.second->avg() * 1000.0) + ", " +
             std::to_string(it.second->this_time() * 1000.0) + "\n";
    }
  }
  return str;
}

void PrintAllTimingStatistics(int print_level) {
  // // Clear Print Time Statistics last time
  // printf("\033[1;1H");
  printf(
      "\nTiming Statistics [total time/count = avg. time, this time (ms)]:\n");
  for (const auto &it : timers) {
    if (it.first.find_first_not_of(' ') < print_level) {
      printf("%s:\t%.3f/%d = %.3f, %.3f\n", it.first.c_str(),
             it.second->sum() * 1000.0, it.second->times(),
             it.second->avg() * 1000.0, it.second->this_time() * 1000.0);
    }
  }
}

void Reset() {
  for (auto &timer : timers) {
    timer.second->reset();
  }
}

void PrintLog(int print_level) { PrintAllTimingStatistics(print_level); }

void SaveLog(const std::string &_save_path) {
  std::ofstream timing_file(_save_path);
  timing_file << llog::GetAllTimingStatistics();
  timing_file.close();
  printf("\033[1;32mTiming Statistics saved to %s\n\033[0m",
         _save_path.c_str());
}
} // namespace llog