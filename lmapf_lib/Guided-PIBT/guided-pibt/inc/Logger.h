#pragma once
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

namespace logging = boost::log;
namespace keywords = boost::log::keywords;



class Logger
{
public:
    Logger(){init();};
    ~Logger(){};

    void set_logfile(std::string filename);
    void init();

    void set_level(logging::trivial::severity_level level) {
        logging::core::get()->set_filter(logging::trivial::severity >= level);
    }
    void log_info(std::string input);
    void log_info(std::string input, int timestep);
    void log_fatal(std::string input);
    void log_fatal(std::string input, int timestep);
    void log_error(std::string input);
    // void log_preprocessing(bool succ);
    // void log_plan(bool succ,int time);
};
