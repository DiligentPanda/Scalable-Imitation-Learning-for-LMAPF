#include "bshoshany/BS_thread_pool.hpp"
#include <iostream>
#include <mutex>

std::mutex m;


void multiply(const int a, const int b)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int p=a * b;
    const std::lock_guard<std::mutex> lock(m);
    std::cout<<p<<"="<<a<<"*"<<b<<std::endl;
}

int main()
{
    BS::thread_pool pool;
    for (int i=0;i<10;++i)
        for (int j=0;j<10;++j)
            pool.push_task(multiply, i, j);

    pool.wait_for_tasks();

    // std::future<int> my_future = pool.submit(multiply, 6, 7);
    // std::cout << my_future.get() <<std::endl;
}