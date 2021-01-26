#ifndef ABSTRACTCLASSES_H
#define ABSTRACTCLASSES_H

#include <Eigen/Dense>

class MABAlgo
{
public:
    virtual ~MABAlgo() = 0;
    virtual int action() = 0;
    virtual void update(int action, double reward) = 0;
    virtual void reset() = 0;
    MABAlgo(std::string name) : name(name) {}

public:
    std::string name;
};
MABAlgo::~MABAlgo() {};

template<typename X>
class Algo
{
public:
    virtual ~Algo() = 0;
    virtual int action(const X& context) = 0;
    virtual void update(const X& context, int action, double reward) = 0;
    virtual std::vector<double> action_distribution(const X& context)  = 0;
    virtual void reset() = 0;
    virtual double upper_bound()
    {
        return -1;
    };
    Algo(std::string name) : name(name) {}

public:
    std::string name;
};

template<typename X>
Algo<X>::~Algo() {};

//A base algorithm is one that can be cloned
template<typename X>
class BaseAlgo: public Algo<X>
{
public:
    BaseAlgo<X>(std::string name) : Algo<X>(name) {}
    virtual std::unique_ptr<BaseAlgo<X>> clone() const = 0;
    virtual double corral_lr(const int horizon, const int nbases) const = 0;
    virtual double exp3_rate(const int horizon, const int nbases) const = 0;
};

class RewardDistribution
{
public:
    virtual double sample() = 0;
    virtual double mean() = 0;
};

template<typename X>
class ContRepresentation
{
public:
    virtual ~ContRepresentation() = 0;
    virtual int n_arms() = 0;
    virtual int features_dim() = 0;
    virtual double features_bound() = 0;
    virtual double param_bound() = 0;

    virtual X sample_context() = 0;
    virtual double sample_reward(const X& context, int action) = 0;

    virtual Eigen::VectorXd get_features(const X& context, int action) = 0;

    virtual double optimal_reward(const X& context) = 0;
    virtual double reward(const X& context, int action) = 0;

};

template<typename X>
ContRepresentation<X>::~ContRepresentation() {};
#endif
