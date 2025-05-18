#pragma once
#include <chrono>
#include <iostream>

using std::chrono::steady_clock;

class TimeLimiter {
public:
    steady_clock::time_point start_time;
    double time_limit;
    bool enabled;

    TimeLimiter(double _time_limit): time_limit(_time_limit), enabled(true) {
        reset_start_time();
    }

    TimeLimiter(const TimeLimiter & other) {
        time_limit = other.time_limit;
        start_time = other.start_time;
    }

    inline void reset_start_time() {
        start_time = steady_clock::now();
    }

    inline double get_elapse() const {
        auto now = steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time);
        return duration.count();
    }

    inline bool timeout() const {
        if (!enabled) return false;

        double elapse=get_elapse();
        return elapse >= time_limit;
    }

    inline double get_remaining_time() const {
        return time_limit - get_elapse();
    }

    inline void disable() {
        enabled = false;
    }

    inline void enable() {
        enabled = true;
    }
};