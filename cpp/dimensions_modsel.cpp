// standard includes and external libraries
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <ctime>
// project includes
#include "abstractclasses.h"
#include "bandit.h"
#include "oful.h"
#include "mmoful.h"
#include "regbalancing.h"
#include "adversarial_master.h"
#include "finitelinrep.h"
#include "utils.h"
#include "gzip.h"

using json = nlohmann::json;
using namespace std;
using namespace Eigen;

size_t PREC = 4;   // for saving numbers are rounded to PREC decimals
size_t EVERY = 1;  // save EVERY round

int main()
{
    std::time_t t = std::time(nullptr);
    char MY_TIME[100];
    std::strftime(MY_TIME, sizeof(MY_TIME), "%Y%m%d%H%M%S", std::localtime(&t));
    std::cout << MY_TIME << '\n';

    typedef std::vector<std::vector<double>> vec2double;

    int seed = time(NULL);
    srand (seed);
    cout << "seed: " << seed << endl;
    // rng.seed(10000); // warm it up
    int n_runs = 6, T = 10000;
    double delta = 0.01;
    double reg_val = 1.;
    double noise_std = 0.3;
    double bonus_scale = 1.;
    bool adaptive_ci = true;
    bool update_all = true;
    //double cor_gamma = 1./T, cor_beta = exp(1./log(T)), cor_lr0 = 20. / sqrt(T);

    std::vector<long> seeds(n_runs);
    std::generate(seeds.begin(), seeds.end(), [] ()
    {
        return rand();
    });

    // FiniteLinearRepresentation rep = make_random(20, 5, 6, true, noise_std, seed);
    // rep.save("linrep.json"); // save current model

    FiniteLinearRepresentation rep("../../problem_data/linrep3.json");

    std::vector<FiniteLinearRepresentation> reps;
    int dim = rep.features_dim();
    double MMM = rep.features_bound();
    for(int i = 2; i <= dim; ++i)
    {
        FiniteLinearRepresentation rr = reduce_dim(rep, i, false, true, true);
        rr = derank_hls(rr, 1, false, true, true);
        //std::cout << rr.features_dim() << std::endl;
        // cout << i << ": " << rr.features_bound() << ", " << rr.param_bound() << std::endl;
        // rr.normalize_features(MMM);
        // cout << i << ": " << rr.features_bound() << ", " << rr.param_bound() << std::endl;
        reps.push_back(rr);
        bool flag = rr.is_equal(rep);
        if (!flag) {
            std::cout << "Error: non realizable representation" << std::endl;
            exit(1);
        }
    }
    assert(rep.is_equal(rep));
    assert(rep.is_hls());
    reps.push_back(rep);
    // cout << dim << ": " << rep.features_bound() << ", " << rep.param_bound() << std::endl;
    //assert(reps.size() == dim);


  //MMOFUL
    vec2double regrets(n_runs), pseudo_regrets(n_runs);
    #pragma omp parallel for
    for (int i = 0; i < n_runs; ++i)
    {
        std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
        for(auto& ll : reps)
        {
            auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
            lreps.push_back(tmp);
        }
        MMOFUL<int> localg(lreps, reg_val, noise_std, bonus_scale, delta/lreps.size(), adaptive_ci);
        ContBanditProblem<int> prb(*lreps[0], localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
        // delete localg;
    }
    // save_vector_csv(regrets, "MMOFUL_regrets.csv", EVERY, PREC);
    save_vector_csv_gzip(regrets, "MMOFUL_regrets.csv.gz", EVERY, PREC);
    // save_vector_csv(pseudo_regrets, "MMOFUL_pseudoregrets.csv", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "MMOFUL_pseudoregrets.csv.gz", EVERY, PREC);


    //EXP3.P
     #pragma omp parallel for
     for (int i = 0; i < n_runs; ++i)
     {
         std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
         std::vector<std::shared_ptr<SmoothedAlgo<int>>> base_algs;
         for(auto& ll : reps)
         {
             auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
             lreps.push_back(tmp);
             base_algs.push_back(
                     std::make_shared<SmoothedAlgo<int>>(
                        SmoothedAlgo<int>(
                            std::make_shared<OFUL<int>>(
                                OFUL<int>(*tmp, reg_val,noise_std,bonus_scale,delta,adaptive_ci)
                            ), seeds[i]
                        )
                     )
             );
         }
         EXP3dotP<int> localg(base_algs, base_algs[0]->base()->exp3_rate(T, base_algs.size()), seeds[i], update_all);
         ContBanditProblem<int> prb(*lreps[0], localg);
         prb.reset();
         prb.run(T);
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
     }
     save_vector_csv_gzip(regrets, "EXP3dotP_regrets.csv.gz", EVERY, PREC);
     save_vector_csv_gzip(pseudo_regrets, "EXP3dotP_pseudoregrets.csv.gz", EVERY, PREC);


    //Corral
     #pragma omp parallel for
     for (int i = 0; i < n_runs; ++i)
     {
         std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
         std::vector<std::shared_ptr<SmoothedAlgo<int>>> base_algs;
         for(auto& ll : reps)
         {
             auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
             lreps.push_back(tmp);
             base_algs.push_back(
                     std::make_shared<SmoothedAlgo<int>>(
                        SmoothedAlgo<int>(
                            std::make_shared<OFUL<int>>(
                                OFUL<int>(*tmp, reg_val,noise_std,bonus_scale,delta,adaptive_ci)
                            ), seeds[i]
                        )
                     )
             );
         }
         Corral<int> localg(base_algs, base_algs[0]->base()->corral_lr(T, base_algs.size()), seeds[i], update_all);
         ContBanditProblem<int> prb(*lreps[0], localg);
         prb.reset();
         prb.run(T);
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
     }
     save_vector_csv_gzip(regrets, "Corral_regrets.csv.gz", EVERY, PREC);
     save_vector_csv_gzip(pseudo_regrets, "Corral_pseudoregrets.csv.gz", EVERY, PREC);

     //EXP4.IX
     #pragma omp parallel for
     for (int i = 0; i < n_runs; ++i)
     {
         std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
         std::vector<std::shared_ptr<Algo<int>>> base_algs;
         for(auto& ll : reps)
         {
             auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
             lreps.push_back(tmp);
             base_algs.push_back(
                     std::make_shared<OFUL<int>>(
                                OFUL<int>(*tmp, reg_val,noise_std,bonus_scale,delta,adaptive_ci)
                            )
             );
         }
         double exp4_gamma = sqrt(2*log(base_algs.size())/(rep.n_arms()*T));
         double exp4_lr = 2*exp4_gamma;
         EXP4dotIX<int> localg(base_algs, exp4_lr, exp4_gamma, seeds[i]);
         ContBanditProblem<int> prb(*lreps[0], localg);
         prb.reset();
         prb.run(T);
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
     }
     save_vector_csv_gzip(regrets, "EXP4dotIX_regrets.csv.gz", EVERY, PREC);
     save_vector_csv_gzip(pseudo_regrets, "EXP4dotIX_pseudoregrets.csv.gz", EVERY, PREC);


     // REGRET Balancing
     #pragma omp parallel for
     for (int i = 0; i < n_runs; ++i)
     {
         std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
         std::vector<std::shared_ptr<Algo<int>>> base_algs;
         for(auto& ll : reps)
         {
             auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
             lreps.push_back(tmp);
             base_algs.push_back(
                 std::make_shared<OFUL<int>>(
                     OFUL<int>(*tmp, reg_val,noise_std,bonus_scale,delta,adaptive_ci)
                 )
             );
         }
         RegretBalance<int> localg(base_algs);
         ContBanditProblem<int> prb(*lreps[0], localg);
         prb.reset();
         prb.run(T);
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
         // delete localg;
     }
     // save_vector_csv(regrets, "OFULBAL_regrets.csv", EVERY, PREC);
     save_vector_csv_gzip(regrets, "OFULBAL_regrets.csv.gz", EVERY, PREC);
     // save_vector_csv(pseudo_regrets, "OFULBAL_pseudoregrets.csv", EVERY, PREC);
     save_vector_csv_gzip(pseudo_regrets, "OFULBAL_pseudoregrets.csv.gz", EVERY, PREC);


     //Regret balancing with elimination
     #pragma omp parallel for
     for (int i = 0; i < n_runs; ++i)
     {
         std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
         std::vector<std::shared_ptr<Algo<int>>> base_algs;
         for(auto& ll : reps)
         {
             auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
             lreps.push_back(tmp);
             base_algs.push_back(
                 std::make_shared<OFUL<int>>(
                     OFUL<int>(*tmp, reg_val,noise_std,bonus_scale,delta,adaptive_ci)
                 )
             );
         }
         RegretBalanceAndEliminate<int> localg(base_algs, delta);
         ContBanditProblem<int> prb(*lreps[0], localg);
         prb.reset();
         prb.run(T);
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
         // delete localg;
     }
     // save_vector_csv(regrets, "OFULBAL_regrets.csv", EVERY, PREC);
     save_vector_csv_gzip(regrets, "OFULBALELIM_regrets.csv.gz", EVERY, PREC);
     // save_vector_csv(pseudo_regrets, "OFULBAL_pseudoregrets.csv", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, "OFULBALELIM_pseudoregrets.csv.gz", EVERY, PREC);

    return 0;
}
