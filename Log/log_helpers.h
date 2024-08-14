#ifndef LOG_HELPERS_H
#define LOG_HELPERS_H
#include <spdlog/spdlog.h>

#define log_trace(...) spdlog::log(spdlog::source_loc(__FILE__, __LINE__, __FUNCTION__), spdlog::level::trace, __VA_ARGS__)
#define log_debug(...) spdlog::log(spdlog::source_loc(__FILE__, __LINE__, __FUNCTION__), spdlog::level::debug, __VA_ARGS__)
#define log_info(...) spdlog::log(spdlog::source_loc(__FILE__, __LINE__, __FUNCTION__), spdlog::level::info, __VA_ARGS__)
#define log_warn(...) spdlog::log(spdlog::source_loc(__FILE__, __LINE__, __FUNCTION__), spdlog::level::warn, __VA_ARGS__)
#define log_error(...) spdlog::log(spdlog::source_loc(__FILE__, __LINE__, __FUNCTION__), spdlog::level::err, __VA_ARGS__)
#define log_critical(...) spdlog::log(spdlog::source_loc(__FILE__, __LINE__, __FUNCTION__), spdlog::level::critical, __VA_ARGS__)

#endif // LOG_HELPERS_H