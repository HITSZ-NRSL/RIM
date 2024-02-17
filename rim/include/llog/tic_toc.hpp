#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>

class TicToc {
public:
  using Ptr = std::shared_ptr<TicToc>;

  TicToc() { reset(); }

  void reset() {
    t_sum = 0.0;
    count = 0;
    tic();
  }

  void tic() { start = std::chrono::high_resolution_clock::now(); }

  double toc() {
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
  }

  double toc_avg() {
    t_this = toc();
    t_sum += t_this;
    ++count;
    return t_sum / (double)count;
  }

  double toc_sum() {
    t_this = toc();
    t_sum += t_this;
    ++count;
    return t_sum;
  }

  void add(const double &t) {
    t_sum += t;
    ++count;
  }

  double avg() const {
    if (count)
      return t_sum / (double)count;
    else
      return 0;
  }
  double sum() const { return t_sum; }
  int times() const { return count; }

  double this_time() const { return t_this; }

  static double now() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double t_sum, t_this;
  int count;
};
