#ifndef BANDIT_H
#define BANDIT_H

#include "abstractclasses.h"
#include <chrono>

#define TIC()                           \
  std::chrono::high_resolution_clock::now();

#define TOC(X)\
std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - X).count() * 1e-9;


class MultiArmedBandit
{
public:
    MultiArmedBandit(std::vector<std::shared_ptr<RewardDistribution>>& rewards, MABAlgo& algo)
    : rewards(rewards), algo(algo)
    {
        reset();
    }

    ~MultiArmedBandit() {}

    void reset()
    {
        t = 0;
        algo.reset();
    }

    void run(int horizon)
    {
        instant_regret.resize(horizon);
        exp_instant_regret.resize(horizon);

        std::vector<double> means;
        double max_mean = std::numeric_limits<double>::min();
        for (int a = 0, aa = rewards.size(); a < aa; ++a){
            means.push_back(rewards[a]->mean());
            max_mean = std::max(means[a], max_mean);
        }

        while (t < horizon)
        {
            auto start = TIC();
            int action = algo.action();
            auto tot = TOC(start);
            std::cout << tot << std::endl;
            start = TIC();
            double reward = rewards[action]->sample();
            tot = TOC(start);
            std::cout << tot << std::endl;
            start = TIC();
            algo.update(action, reward);
            tot = TOC(start);
            std::cout << " ------------ \n" << std::endl;

            //compute regret
            exp_instant_regret[t] = max_mean - means[action];
            instant_regret[t] =  max_mean - reward;

            ++t;
        }
    }

public:
    std::vector<std::shared_ptr<RewardDistribution>>& rewards;
    MABAlgo& algo;
    int t;
    std::vector<double> instant_regret;
    std::vector<double> exp_instant_regret;
};

template<typename X>
class ContBanditProblem
{
public:
    ContBanditProblem(ContRepresentation<X>& reward_rep, Algo<X>& algo)
        : reward_rep(reward_rep), algo(algo), t(0)
    {
        reset();
    }

    void reset()
    {
        t = 0;
        algo.reset();
    }

    void run(int horizon)
    {
        instant_regret.resize(horizon);
        exp_instant_regret.resize(horizon);
        while (t < horizon)
        {
            // auto start = TIC();
            X context = reward_rep.sample_context();
            // auto tot = TOC(start);
            // std::cout << "sample: " << tot << std::endl;
            // start = TIC();
            int action = algo.action(context);
            // tot = TOC(start);
            // std::cout << "action: " << tot << std::endl;
            // start = TIC();
            double reward = reward_rep.sample_reward(context, action);
            // tot = TOC(start);
            // std::cout << "reward: " << tot << std::endl;
            // start = TIC();
            algo.update(context, action, reward);
            // tot = TOC(start);
            // std::cout << "update: " << tot << std::endl;
            // std::cout << " ------------ \n" << std::endl;

            //compute regret
            double opt_exprew = reward_rep.optimal_reward(context);
            double exp_rew = reward_rep.reward(context, action);
            exp_instant_regret[t] = opt_exprew - exp_rew;
            instant_regret[t] =  opt_exprew - reward;

            ++t;
        }
    }

public:
    ContRepresentation<X>& reward_rep;
    Algo<X>& algo;
    int t;
    std::vector<double> instant_regret;
    std::vector<double> exp_instant_regret;
};

#endif
