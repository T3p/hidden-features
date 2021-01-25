#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include<Eigen/Dense>
#include<fstream>
#include<vector>
#include<sstream>
#include "gzip.h"

using namespace std;
using namespace Eigen;

template<typename T>
void print_vector(std::vector<T> v)
{
    for(T& h : v)
    {
        std::cout << h << " ";
    }
    std::cout << std::endl;
}

template <class X>
void save_vector_csv(
    std::vector<std::vector<X>> v, std::string name,
    size_t every = 1, size_t precision = 10
)
{
    every = max(every, (size_t) 1);
    size_t T = v[0].size();
    size_t m = v.size();
    std::ofstream file(name);
    if (file.is_open())
    {
        file.precision(precision);
        for (size_t t = 0; t < T; t += every)
        {
            for (size_t j = 0; j < m; ++j)
            {
                file << v[j][t];
                if (j < m - 1)
                {
                    file << ",";
                }
            }
            file << endl;
        }
    }
}
template <class X>
void save_vector_csv_gzip(
    std::vector<std::vector<X>> v, std::string name,
    size_t every = 1, size_t precision = 10
)
{
    every = max(every, (size_t) 1);
    size_t T = v[0].size();
    size_t m = v.size();
    std::ofstream file(name);
    if (file.is_open())
    {
        std::stringstream clear;
        clear.precision(precision);
        for (size_t t = 0; t < T; t += every)
        {
            for (size_t j = 0; j < m; ++j)
            {
                clear << v[j][t];
                if (j < m - 1)
                {
                    clear << ",";
                }
            }
            clear << endl;
        }
        std::string comp = Gzip::compress(clear.str());
        file << comp;
    }
}

//check probabilities
bool is_distr(const vector<double>& probs, double epsilon=1e-6)
{
    bool cond1 = all_of(probs.begin(), probs.end(), [](double p){return p>=0.;});
    bool cond2 = (abs(accumulate(probs.begin(), probs.end(), 0.) - 1.) < epsilon);
    return cond1 && cond2;
}

#endif
