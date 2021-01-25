#ifndef UCB_H
#define UCB_H

#include <iostream>
#include <vector>
#include <boost/math/tools/minima.hpp>
#include "abstractclasses.h"

class UCB : public MABAlgo
{
public:

    UCB(size_t n_arms, double rrange=1., double scale=1.)
    : MABAlgo("UCB1"), n_arms(n_arms), rrange(rrange), scale(scale)
    {
        reset();
    }

    void reset() {
        R.resize(n_arms);
        N.resize(n_arms);
        std::fill(R.begin(), R.end(), 0);
        std::fill(N.begin(), N.end(), 0);
        UCBindex.resize(n_arms);
        t = 1;
    }

    int action() {
        double max_val = std::numeric_limits<double>::min();
        int action = 0;
        for (size_t i=0; i < n_arms; ++i) {
            UCBindex[i] = std::numeric_limits<double>::max();
            if (N[i] > 0) {
                UCBindex[i] = R[i] / N[i] + scale * rrange * sqrt(.5 * log(t) / N[i]);
            }
            if (max_val < UCBindex[i]) {
                max_val = UCBindex[i];
                action = i;
            }
        }
        return action;
    }

    void update(int action, double reward) {
        R[action] += reward;
        N[action] += 1;
        t++;
    }

public:
    size_t n_arms;
    double rrange, delta, scale;
    std::vector<double> R, UCBindex;
    std::vector<int> N;
    int t;

};

class UCBV : public UCB
{
public:
    UCBV(size_t n_arms, double rrange=1., double scale=1.)
    : UCB(n_arms, rrange, scale)
    {
        name = "UCBV";
    }

    void reset() {
        UCB::reset();
        M.resize(n_arms);
        std::fill(M.begin(), M.end(), 0);
    }

    int action() {
        double max_val = std::numeric_limits<double>::min();
        int action = 0;
        for (size_t i=0; i < n_arms; ++i) {
            UCBindex[i] = std::numeric_limits<double>::max();
            if (N[i] > 0) {
                double var = M[i] / N[i];
                UCBindex[i] = R[i] / N[i] + scale * sqrt(2. * var * log(t) / N[i]) + scale * 8. * rrange * log(t) / (3. * N[i]);
            }
            if (max_val < UCBindex[i]) {
                max_val = UCBindex[i];
                action = i;
            }
        }
        return action;
    }

    void update(int action, double reward) {
        double old_mean = N[action] > 0 ? R[action] / N[action] : 0;
        R[action] += reward;
        N[action] += 1;
        t++;
        double mean = R[action] / N[action];
        M[action] += (reward - old_mean) * (reward - mean); 
    }
public: 
    std::vector<double> M;
};


class MOSS : public UCB
{

public:
    MOSS(size_t n_arms, double rrange=1., double scale=1.)
    : UCB(n_arms, rrange, scale)
    {
        name = "MOSS";
    }

    int action() {
        double max_val = std::numeric_limits<double>::min();
        int action = 0;
        for (size_t i=0; i < n_arms; ++i) {
            UCBindex[i] = std::numeric_limits<double>::max();
            if (N[i] > 0) {
                UCBindex[i] = R[i] / N[i] + scale * sqrt(std::max(log(t/(1. * n_arms * N[i])), 0.) / N[i]);
            }
            if (max_val < UCBindex[i]) {
                max_val = UCBindex[i];
                action = i;
            }
        }
        return action;
    }
};

// class BernKLUCB : public UCB
// {
// public:
//     int action() {
//         int bits = std::numeric_limits<double>::digits;
//         double max_val = std::numeric_limits<double>::min();
//         int action = 0;
//         for (size_t i=0; i < n_arms; ++i) {
//             UCBindex[i] = std::numeric_limits<double>::max();
//             if (N[i] > 0) {
//                 std::pair<double, double> result = boost::math::tools::brent_find_minima(
//                             [N, R, i](double const& x){
//                                 double p = R[i]/N[i], q = x;
//                                 double klber = p * log(p/q) + (1.-p) * log((1.-p)/(1.-q));
//                                 double v = N[i] * klber - log(t);
//                                 return v*v;
//                             },
//                             0, 1, bits
//                         );
//             }
//             if (max_val < UCBindex[i]) {
//                 max_val = UCBindex[i];
//                 action = i;
//             }
//         }
//         return action;
//     }
// };

#endif