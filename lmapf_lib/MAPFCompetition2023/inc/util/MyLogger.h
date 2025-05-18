#pragma once
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <string>
#include <ctime>
// #include "spdlog/fmt/bundled/color.h"
#include "util/Dev.h"

class MyLogger {
public:
    std::string name;
    std::shared_ptr<spdlog::logger> logger;

    void init(std::string log_fp_prefix, std::string name="g", spdlog::level::level_enum level=spdlog::level::debug) {
        this->name=name;

        // get timestamp
        std::string timestamp=get_timestep();
        std::string log_fp=log_fp_prefix+"_"+timestamp+".log";

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        // console_sink->set_level(spdlog::level::debug);
        // console_sink->set_pattern("[multi_sink_example] [%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_fp, true);
        // file_sink->set_level(spdlog::level::trace);

        logger=std::make_shared<spdlog::logger>(name, spdlog::sinks_init_list({console_sink, file_sink}));
        logger->set_level(level);

        // logger->debug(
        //     fmt::format(
        //         // fmt::bg( fmt::terminal_color::yellow ) |
        //         fmt::fg( fmt::terminal_color::cyan ), //|
        //         // fmt::emphasis::bold,
        //         "{}_logger initialized",
        //         name
        //     )
        // );
    }

    // get timestep
    static std::string get_timestep(){
        auto now=time(0);
        auto _time = localtime(&now);
        char buffer[50];

        strftime(buffer,50,"%y-%m-%d %H:%M:%S",_time);
        
        return std::string(buffer);
    }

    void set_level(spdlog::level::level_enum level=spdlog::level::debug){
        logger->set_level(level);
    }

    template<typename... Args>
    void debug(std::string msg, const Args & ... args) {
        logger->debug(msg, args...);
    }

    template<typename... Args>
    void info(std::string msg, const Args & ... args) {
        logger->info(msg, args...);
    }

    template<typename... Args>
    void warn(std::string msg, const Args & ... args) {
        logger->warn(msg, args...);
    }

    template<typename... Args>
    void error(std::string msg, const Args & ... args) {
        logger->error(msg, args...);
    }

    void flush() {
        logger->flush();
    }

    void flush_on(spdlog::level::level_enum level=spdlog::level::err) {
        logger->flush_on(level);
    }

    void flush_every(std::chrono::seconds interval=std::chrono::seconds(5)) {
        spdlog::flush_every(interval);
    }

};

extern MyLogger g_logger;

#define DEV_DEBUG(msg, args...) ONLYDEV(g_logger.debug(msg, ##args);)
#define DEV_INFO(msg, args...) ONLYDEV(g_logger.info(msg, ##args);)
#define DEV_WARN(msg, args...) ONLYDEV(g_logger.warn(msg, ##args);)
#define DEV_ERROR(msg, args...) ONLYDEV(g_logger.error(msg, ##args);)