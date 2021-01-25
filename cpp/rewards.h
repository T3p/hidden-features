#ifndef REWARDS_H
#define REWARDS_H

#include "abstractclasses.h"
#include <random>


class Bernoulli : public RewardDistribution
{
public:
    Bernoulli(double p, int seed)
    : p(p), seed(seed)
    {
        dist = std::bernoulli_distribution(p);
        gen = std::mt19937(seed);
    }

    double sample() {
        return dist(gen);
    }

    double mean() {
        return p;
    }

public:
    double p;
    std::bernoulli_distribution dist;
    int seed;
    std::mt19937 gen;
};

#endif