#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include "oful.h"
#include "contlinrep.h"
#include "utils.h"
#include "gzip.h"
#include "bandit.h"

using json = nlohmann::json;
using namespace std;
using namespace Eigen;

int PREC = 4;
int EVERY = 1;

int main()
{
    std::time_t t = std::time(nullptr);
    char MY_TIME[100];
    std::strftime(MY_TIME, sizeof(MY_TIME), "%Y%m%d%H%M%S", std::localtime(&t));
    std::cout << MY_TIME << '\n';

    typedef std::vector<std::vector<double>> vec2double;

    srand (time(NULL));
    // rng.seed(10000); // warm it up
    int n_runs = 50, T = 20000;
    double delta = 0.01;
    double reg_val = 1.;
    double noise_std = 0.2;
    double bonus_scale = 1;
    bool adaptive_ci = true;

    std::vector<long> seeds(n_runs);
    std::generate(seeds.begin(), seeds.end(), [] ()
    {
        return rand();
    });

    vec2double regrets(n_runs), pseudo_regrets(n_runs);
    #pragma omp parallel for
    for (int i = 0; i < n_runs; ++i)
    {
        ContToy1_phi1 lrep = ContToy1_phi1(noise_std, seeds[i]);
        OFUL<std::vector<double>> localg(lrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
        ContBanditProblem<std::vector<double>> prb(lrep, localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, "OFULCont1_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "OFULCont1_pseudoregrets.csv.gz", EVERY, PREC);

    #pragma omp parallel for
    for (int i = 0; i < n_runs; ++i)
    {
        ContToy1_phi2 lrep = ContToy1_phi2(noise_std, seeds[i]);
        OFUL<std::vector<double>> localg(lrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
        ContBanditProblem<std::vector<double>> prb(lrep, localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, "OFULCont2_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "OFULCont2_pseudoregrets.csv.gz", EVERY, PREC);

    return 0;
}
