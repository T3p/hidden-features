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
#include "leader.h"
#include "adversarial_master.h"


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
    int n_runs = 20, T = 1000;
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

    ContToy1_phi1 lrep1 = ContToy1_phi1(noise_std, seeds[0]);
    ContToy1_phi2 lrep2 = ContToy1_phi2(noise_std, seeds[0]);

    std::cerr << "OFUL" << std::endl;
    for (int i = 0; i < n_runs; ++i)
    {
        OFUL<std::vector<double>> localg(lrep1, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
        ContBanditProblem<std::vector<double>> prb(lrep1, localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, "$\\phi#1$_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "$\\phi#1$_pseudoregrets.csv.gz", EVERY, PREC);

    for (int i = 0; i < n_runs; ++i)
    {
        OFUL<std::vector<double>> localg(lrep2, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
        ContBanditProblem<std::vector<double>> prb(lrep2, localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, "$\\phi#2$_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "$\\phi#2$_pseudoregrets.csv.gz", EVERY, PREC);

    std::vector<std::shared_ptr<ContRepresentation<std::vector<double>>>> lreps;

    auto tmp = std::make_shared<ContToy1_phi1>(lrep1);
    lreps.push_back(tmp);
    tmp = std::make_shared<ContToy1_phi2>(lrep2);
    lreps.push_back(tmp);

    //LEADER
    std::cerr << "LEADER" << std::endl;
    for (int i = 0; i < n_runs; ++i)
    {
        LEADER<std::vector<double>> localg(lreps, reg_val, noise_std, bonus_scale, delta/lreps.size(), adaptive_ci);
        ContBanditProblem<std::vector<double>> prb(*lreps[0], localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
        // delete localg;
    }
    save_vector_csv_gzip(regrets, "\\algo_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "\\algo_pseudoregrets.csv.gz", EVERY, PREC);

    //EXP4.IX
    std::cerr << "EXP4" << std::endl;
    for (int i = 0; i < n_runs; ++i)
    {
        std::vector<std::shared_ptr<Algo<std::vector<double>>>> base_algs;
        for(auto& ll : lreps)
        {
            base_algs.push_back(
                std::make_shared<OFUL<std::vector<double>>>(
                    OFUL<std::vector<double>>(*tmp, reg_val,noise_std,bonus_scale,delta,adaptive_ci)
                )
            );
        }
        double exp4_gamma = sqrt(2*log(base_algs.size())/(4*T));
        double exp4_lr = 2*exp4_gamma;
        EXP4dotIX<std::vector<double>> localg(base_algs, exp4_lr, exp4_gamma, seeds[i]);
        ContBanditProblem<std::vector<double>> prb(*lreps[0], localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, "\\expfour_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "\\expfour_pseudoregrets.csv.gz", EVERY, PREC);

    return 0;
}
