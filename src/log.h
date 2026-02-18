#pragma once
/// log.h — Structured logging with compile-time level filtering.
///
/// Usage:
///   VT_INFO("Selected GPU: " << name);
///   VT_DEBUG("Buffer size: " << size << " bytes");
///   vultorch::log_level() = vultorch::LogLevel::Quiet;  // silence all

#include <iostream>

namespace vultorch {

enum class LogLevel { Quiet = 0, Error = 1, Warn = 2, Info = 3, Debug = 4 };

inline LogLevel& log_level() {
    static LogLevel lvl = LogLevel::Info;
    return lvl;
}

} // namespace vultorch

#define VT_LOG(level, ...)                                                      \
    do {                                                                        \
        if (static_cast<int>(level) <=                                          \
            static_cast<int>(vultorch::log_level())) {                          \
            std::cout << "[vultorch] " << __VA_ARGS__ << "\n";                  \
        }                                                                       \
    } while (0)

#define VT_INFO(...)  VT_LOG(vultorch::LogLevel::Info,  __VA_ARGS__)
#define VT_WARN(...)  VT_LOG(vultorch::LogLevel::Warn,  __VA_ARGS__)
#define VT_ERROR(...) VT_LOG(vultorch::LogLevel::Error, __VA_ARGS__)
#define VT_DEBUG(...) VT_LOG(vultorch::LogLevel::Debug, __VA_ARGS__)
