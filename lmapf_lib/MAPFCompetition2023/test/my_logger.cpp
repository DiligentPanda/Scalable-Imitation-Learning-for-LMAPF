#include "util/MyLogger.h"
#include <spdlog/fmt/bundled/color.h>

int main() {
    g_logger.init("logs/test");
    
    g_logger.error("this is error");
    g_logger.warn("this is warn");
    g_logger.info("this is info");
    DEV_DEBUG("this is debug");

    g_logger.info("this is simple formatted info: hello, {}", "world");
    g_logger.info("this is colored formatted info: hello, {}", "world");

    
    g_logger.info( "this is colored formatted info: hello, {}",
        fmt::format(
            // fmt::bg( fmt::terminal_color::yellow ) |
            fmt::fg( fmt::terminal_color::yellow ) |
            fmt::emphasis::bold,
            "world"
        )
    );

    DEV_DEBUG("this is macro debug: hello, world");
    DEV_ERROR( "this is colored formatted macro error: hello, {}",
        fmt::format(
            // fmt::bg( fmt::terminal_color::yellow ) |
            fmt::fg( fmt::terminal_color::yellow ) |
            fmt::emphasis::bold,
            "world"
        )
    )
}