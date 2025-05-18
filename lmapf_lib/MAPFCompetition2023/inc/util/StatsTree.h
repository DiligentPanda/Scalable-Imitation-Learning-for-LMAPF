#pragma once
#include <cstring>

// TODO(rivers): if building takes too much time, consider using 2D semengtaion tree
// https://usaco.guide/plat/2DRQ?lang=cpp
struct StatsTree {
    int * row_cum_sum;
    int * col_cum_sum;   
    int h;
    int w;

    StatsTree(int w, int h): h(h), w(w) {
        row_cum_sum = new int[h*(w+1)];
        col_cum_sum = new int[w*(h+1)];
        clear();
        // memset(col_cum_sum, 0, sizeof(int)*w*(h+1));
    }

    ~StatsTree() {
        delete[] row_cum_sum;
        delete[] col_cum_sum;
    }

    void clear() {
        memset(row_cum_sum, 0, sizeof(int)*h*(w+1));
        memset(col_cum_sum, 0, sizeof(int)*w*(h+1));        
    }

    void update(int pos, int val=1) {
        int x = pos % w;
        int y = pos / w;
        update(x, y, val);
    }

    void update(int x, int y, int val) {
        // update row y: start from x+1, which count the previous x grids sum.
        for (int i=x+1; i<=w; ++i) {
            row_cum_sum[y*(w+1)+i] += val;
        }

        // update col x: start form y+1
        for (int j=y+1; j<=h; ++j) {
            col_cum_sum[x*(h+1)+j] += val;
        }
    }

    // return sum from grids c1 to c2-1
    int query_row(int r, int c1, int c2) {
        return row_cum_sum[r*(w+1)+c2] - row_cum_sum[r*(w+1)+c1];
    }

    // return sum from grids r1 to r2-1
    int query_col(int c, int r1, int r2) {
        return col_cum_sum[c*(h+1)+r2] - col_cum_sum[c*(h+1)+r1];
    }

};

struct StatsTrees {
    StatsTree obstacle_stats_tree;
    StatsTree agent_stats_tree;

    StatsTrees(int w, int h): obstacle_stats_tree(w, h), agent_stats_tree(w, h) {}

    void reset_agent_stats_tree() {
        agent_stats_tree.clear();
    }

    void query(int s_x, int s_y, int g_x, int g_y) {
        // search for row break point
        int delta_y;
        if (s_y < g_y) {
            delta_y = 1;
        } else {
            delta_y = -1;
        }



    }
};
