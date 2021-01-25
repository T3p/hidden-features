#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <chrono>
#include <memory>
#include "bandit.h"
#include "ucb.h"
#include "rewards.h"
#include "utils.h"

using namespace std;
using namespace Eigen;

#define TIC()                           \
  chrono::high_resolution_clock::now();

#define TOC(X)\
chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - X).count() * 1e-9;


int PREC = 4;
int EVERY = 1;

std::vector<std::shared_ptr<RewardDistribution>> make_problem(std::vector<double> p, int seed) {
    std::vector<std::shared_ptr<RewardDistribution>> vec;
    for(auto& y : p) {
        vec.push_back(std::make_shared<Bernoulli>(Bernoulli(y, seed)));
    }
    return vec;
}

int main()
{
    std::time_t t = std::time(nullptr);
    char MY_TIME[100];
    std::strftime(MY_TIME, sizeof(MY_TIME), "%Y%m%d%H%M%S", std::localtime(&t));
    std::cout << MY_TIME << '\n';

    typedef std::vector<std::vector<double>> vec2double;

    int seed = time(NULL);
    cout << "seed: " << seed << endl;
    srand (seed);
    int n_runs = 200, T = 600;
    double rrange = 1.;
    double bonus_scale = 1.;
    int n_arms = 4;

    std::vector<long> seeds(n_runs);
    std::generate(seeds.begin(), seeds.end(), [] ()
    {
        return rand();
    });
    
    /// define rewards
    cout << "Bernoulli( ";
    std::vector<double> means;
    for (int i=0; i < n_arms; ++i) {
        double p = ((double) rand() / (RAND_MAX));
        cout << p << ", ";
        means.push_back(p);
    }
    cout << ")" << endl;
    ///

    vec2double regrets(n_runs), pseudo_regrets(n_runs);
    #pragma omp parallel for
    for (int i = 0; i < n_runs; ++i)
    {
        std::vector<std::shared_ptr<RewardDistribution>> rews = make_problem(means, seeds[i]);
        UCB localg(rews.size(), rrange, bonus_scale);
        MultiArmedBandit prb(rews, localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, "UCBV_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "UCBV_pseudoregrets.csv.gz", EVERY, PREC);

    for(auto& y : means) {
        cout << y << ", ";
    }
    cout << endl;
    return 0;
}
