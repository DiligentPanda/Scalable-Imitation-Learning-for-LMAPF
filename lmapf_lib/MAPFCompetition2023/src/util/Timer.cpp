#include "util/Timer.h"

Timer g_timer;

steady_clock::time_point Timer::record_p(const string & pkey)
{
    auto t=steady_clock::now();
    time_points[pkey]=t;
    return t;
}

steady_clock::time_point Timer::get_p(const string & pkey) const
{
    const auto & iter=time_points.find(pkey);
    if (iter!=time_points.end())
    {
        return iter->second;
    } else 
    {
        // error
        cerr<<"key "<<pkey<<" not found in time_points"<<endl;
        exit(-1);
    }
}


double Timer::record_d(const string & old_pkey, const string & new_pkey, const string & dkey)
{
    auto new_p=record_p(new_pkey);
    auto old_p=get_p(old_pkey);
    auto d=duration<double>(new_p-old_p).count();

    const auto & iter=time_durations.find(dkey);
    if (iter!=time_durations.end())
    {
        time_durations[dkey]+=d;
        time_duration_counters[dkey]++;
        time_durations_last[dkey]=d;
    }
    else
    {
        time_durations[dkey]=d;
        time_duration_counters[dkey]=1;
        time_durations_last[dkey]=d;
    }
    return d;
}

double Timer::record_d(const string & old_pkey, const string & new_pkey)
{
    return record_d(old_pkey,new_pkey,new_pkey);
}


double Timer::get_d(const string & dkey, int mode) const
{
    const auto & iter=time_durations.find(dkey);
    double d=0;
    if (iter!=time_durations.end())
    {
        d=iter->second;
    }
    else
    {
        // error
        cerr<<"key "<<dkey<<" not found in time_durations"<<endl;
        exit(-1);
    }   
    double ret=d;
    if (mode==1)
    {
        ret=ret/double(time_duration_counters.at(dkey));
    }
    if (mode==2) 
    {
        return time_durations_last.at(dkey);
    }
    return ret;
}

std::unordered_map<string,double> Timer::get_all_d(int mode) const
{
    std::unordered_map<string,double> ret;
    for (const auto & pair: time_durations)
    {
        ret[pair.first]=get_d(pair.first,mode);
    }
    return ret;
}

void Timer::remove_p(const string & pkey)
{
    int num=time_points.erase(pkey);
    if (num==0)
    {
        // error
        cerr<<"key "<<pkey<<" not found in time_points"<<endl;
        exit(-1);
    }
}

void Timer::remove_d(const string & dkey)
{
    int num=time_durations.erase(dkey);
    if (num==0)
    {
        // error
        cerr<<"key "<<dkey<<" not found in time_durations"<<endl;
        exit(-1);
    }
    time_duration_counters.erase(dkey);
    time_durations_last.erase(dkey);
}

void Timer::clear_d()
{
    time_durations.clear();
    time_duration_counters.clear();
    time_durations_last.clear();
}

void Timer::clear_p()
{
    time_points.clear();
}

void Timer::clear()
{
    clear_p();
    clear_d();
}

void Timer::print_d(const string & dkey) const
{
    double sum_d=get_d(dkey,0);
    double mean_d=get_d(dkey,1);
    double last_d=get_d(dkey,2);
    DEV_DEBUG("the time cost of {} is sum: {:.3f}, mean: {:.3f}, last:{:.3f}.",
        dkey,
        sum_d,
        mean_d,
        last_d
    );
}

void Timer::print_all_d() const
{
    for (const auto & pair: time_durations)
    {
        print_d(pair.first);
    }
}