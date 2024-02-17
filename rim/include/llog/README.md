
```
static auto p_timer = llog::CreateTimer("get_input");
p_timer->tic();
p_timer->toc_sum();

llog::PrintAllTimingStatistics
llog::PrintLog();
```