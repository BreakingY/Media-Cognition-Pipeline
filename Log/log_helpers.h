#ifndef LOG_HELPERS_H
#define LOG_HELPERS_H
#include <spdlog/spdlog.h>


#define log_trace(...) spdlog::trace(__VA_ARGS__)
#define log_debug(...) spdlog::debug(__VA_ARGS__)
#define log_info(...) spdlog::info(__VA_ARGS__)
#define log_warn(...) spdlog::warn(__VA_ARGS__)
#define log_error(...) spdlog::error(__VA_ARGS__)
#define log_critical(...) spdlog::critical(__VA_ARGS__)
#endif // LOG_HELPERS_H