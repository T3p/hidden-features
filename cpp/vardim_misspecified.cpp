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
#include "leaderelim.h"
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
    int n_runs = 6, T = 80000;
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

    FiniteLinearRepresentation reference_rep = flr_loadjson("../../problem_data/linrep3.json", noise_std, seed);

    //CREATE CANDIDATE REPRESENTATIONS
    std::vector<FiniteLinearRepresentation> reps;
    int dim = reference_rep.features_dim();
    double MMM = reference_rep.features_bound();
    for(int i = 2; i <= dim; ++i)
    {
        FiniteLinearRepresentation rr = reduce_dim(reference_rep, i, false, true, true);
        rr = derank_hls(rr, 1, false, true, true);
        assert(!rr.is_hls());
        assert(rr.features_dim()==i);
        bool flag = rr.is_equal(reference_rep);
        if (!flag)
        {
            std::cout << "Error: non realizable representation" << std::endl;
            exit(1);
        }
        reps.push_back(rr);
    }
    assert(reference_rep.is_equal(reference_rep));
    assert(reference_rep.is_hls());
    reps.push_back(reference_rep);
    reps.push_back(make_reshaped_linrep(reference_rep, dim/2, rand()));
    reps.push_back(make_reshaped_linrep(reference_rep, dim/3, rand()));
    reps.push_back(make_random(reference_rep.n_contexts(), reference_rep.n_arms(), 9, true, noise_std, rand()));
    reps.push_back(make_random(reference_rep.n_contexts(), reference_rep.n_arms(), 3, true, noise_std, rand()));



    //OFUL VS LEADER
    path = "";
    // path = "../../logs/vardim/oful/";

    //just OFUL
    // for(int j = 0; j < reps.size(); ++j)
    // {
    //     vec2double regrets(n_runs), pseudo_regrets(n_runs);

    //     #pragma omp parallel for
    //     for (int i = 0; i < n_runs; ++i)
    //     {
    //         FiniteLinearRepresentation lrep = reps[j].copy(seeds[i]);
    //         OFUL<int> localg(lrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
    //         FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);
    //         ContBanditProblem<int> prb(cpRefRep, localg);
    //         prb.reset();
    //         prb.run(T);
    //         regrets[i] = prb.instant_regret;
    //         pseudo_regrets[i] = prb.exp_instant_regret;
    //     }
    //     name = "$d="+std::to_string(reps[j].features_dim()) + "$" + (reps[j].is_hls()? " (HLS)" : "");
    //     std::cout << name << std::endl;
    //     save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    //     save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);
    // }

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
        ElimLEADER<int> localg(lreps, reg_val, noise_std, bonus_scale, delta/lreps.size(), adaptive_ci);
        FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);
        ContBanditProblem<int> prb(cpRefRep, localg);
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
    path = "";
    // path = "../../logs/vardim/modsel/";

    //save LEADER also in model selection dim
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
        FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);
        ContBanditProblem<int> prb(cpRefRep, localg);
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
