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
#include "leader.h"
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
    // seed= 1611835201;
    seed=1611836268;
    srand (seed);
    cout << "seed: " << seed << endl;
    // rng.seed(10000); // warm it up
    int n_runs = 20, T = 10000;
    double delta = 0.01;
    double reg_val = 1.;
    double noise_std = 0.3;
    double bonus_scale = 1.;
    bool adaptive_ci = true;
    bool update_all = true;
    std::string path;
    std::string name;

    std::vector<long> seeds(n_runs);
    std::generate(seeds.begin(), seeds.end(), [] ()
    {
        return rand();
    });

    // FiniteLinearRepresentation rep = make_random(20, 5, 6, true, noise_std, seed);
    // rep.save("linrep.json"); // save current model

    //FiniteLinearRepresentation rep("linrep3.json");
    // FiniteLinearRepresentation rep("../../problem_data/linrep3.json");
    FiniteLinearRepresentation rep = flr_loadjson("../../problem_data/linrep3.json", noise_std, seed);

    //CREATE CANDIDATE REPRESENTATIONS
    std::vector<FiniteLinearRepresentation> reps;
    int dim = rep.features_dim();
    double MMM = rep.features_bound();
    for(int i = 2; i <= dim; ++i)
    {
        FiniteLinearRepresentation rr = reduce_dim(rep, i, false, true, true);
        rr = derank_hls(rr, 1, false, true, true);
        assert(!rr.is_hls());
        assert(rr.features_dim()==i);
        bool flag = rr.is_equal(rep);
        if (!flag)
        {
            std::cout << "Error: non realizable representation" << std::endl;
            exit(1);
        }
        reps.push_back(rr);
    }
    assert(rep.is_equal(rep));
    assert(rep.is_hls());
    reps.push_back(rep);



    //OFUL VS LEADER
    //path = "";
    path = "../../logs/vardim/oful/";

    //just OFUL
    for(int j = 0; j < reps.size(); ++j)
    {
        vec2double regrets(n_runs), pseudo_regrets(n_runs);

        #pragma omp parallel for
        for (int i = 0; i < n_runs; ++i)
        {
            FiniteLinearRepresentation lrep = reps[j].copy(seeds[i]);
            OFUL<int> localg(lrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
            ContBanditProblem<int> prb(lrep, localg);
            prb.reset();
            prb.run(T);
            regrets[i] = prb.instant_regret;
            pseudo_regrets[i] = prb.exp_instant_regret;
        }
        name = "$d="+std::to_string(reps[j].features_dim()) + "$" + (reps[j].is_hls()? " (HLS)" : "");
        std::cout << name << std::endl;
        save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
        save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);
    }

    //LEADER
    name = "\\algo";
    std::cout << name << std::endl;
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
        LEADER<int> localg(lreps, reg_val, noise_std, bonus_scale, delta/lreps.size(), adaptive_ci);
        ContBanditProblem<int> prb(*lreps[0], localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
        // delete localg;
    }
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);



    //return 0;
    //LEADER VS MODEL SELECTION BASELINES
    //path = "";
    path = "../../logs/vardim/modsel/";

    //save LEADER also in model selection dim
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);

    //EXP3.P
    name = "\\expthree";
    std::cout << name << std::endl;
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
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);


    //Corral
    name = "\\corral";
    std::cout << name << std::endl;
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
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);

    //EXP4.IX
    name = "\\expfour";
    std::cout << name << std::endl;
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
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);


    //REGRET Balancing
    name = "\\regbal";
    std::cout << name << std::endl;
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
        RegretBalance<int> localg(base_algs, update_all);
        ContBanditProblem<int> prb(*lreps[0], localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
        // delete localg;
    }
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);


    //Regret balancing with elimination
    name = "\\regbalelim";
    std::cout << name << std::endl;
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
        RegretBalanceAndEliminate<int> localg(base_algs, delta, update_all);
        ContBanditProblem<int> prb(*lreps[0], localg);
        prb.reset();
        prb.run(T);
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
        // delete localg;
    }
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);


    return 0;
}
