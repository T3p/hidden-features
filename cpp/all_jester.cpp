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

    const char *files[7] = { "../../problem_data/jester/33/1/jester_post_d33_span33.npz", "../../problem_data/jester/33/1/jester_post_d26_span26.npz", "../../problem_data/jester/33/1/jester_post_d24_span24.npz", "../../problem_data/jester/33/1/jester_post_d23_span23.npz", "../../problem_data/jester/33/1/jester_post_d20_span20.npz", "../../problem_data/jester/33/1/jester_post_d17_span17.npz", "../../problem_data/jester/33/1/jester_post_d16_span16.npz" };

    const char *names[7] = {"d=33", "d=26", "d=24", "d=23", "d=20", "d=17", "d=16"};

    std::time_t t = std::time(nullptr);
    char MY_TIME[100];
    std::strftime(MY_TIME, sizeof(MY_TIME), "%Y%m%d%H%M%S", std::localtime(&t));
    std::cout << MY_TIME << '\n';

    typedef std::vector<std::vector<double>> vec2double;

    int seed = time(NULL);
    srand (seed);
    cout << "seed: " << seed << endl;
    int n_runs = 1, T = 100;
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

    // load reference representation
    // FiniteLinearRepresentation reference_rep=flr_loadjson("linrep3.json", noise_std, seed);
    FiniteLinearRepresentation reference_rep=flr_loadnpz("./../problem_data/jester/33/1/jester_post_d33_span33.npz", noise_std, seed, "features", "theta");
    cout << "Ref_rep.dim: " << reference_rep.features_dim() << endl;
    cout << "Ref_rep.feat_bound=" << reference_rep.features_bound() << endl;

    // other representations
    std::vector<FiniteLinearRepresentation> reps;
    for (int i = 0; i < 7; ++i) {

	FiniteLinearRepresentation rr = flr_loadnpz(files[i], noise_std, seed, "features", "theta");

        cout << "phi_" << i << ".dim=" << rr.features_dim() << endl;
        cout << "phi_" << i << ".feat_bound=" << rr.features_bound() << endl;
        bool flag = reference_rep.is_equal(rr, 0.05);
        cout << "phi_" << i << ".equal_ref=" << flag << endl;
        if (!flag) {
            std::cout << "Error: " << i << "is a non realizable representation" << std::endl;
            exit(1);
        }
        reps.push_back(rr);
    }

    //LEADER
    name = "LEADER";
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
        // create same representation but witth different seed
        FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);
        ContBanditProblem<int> prb(cpRefRep, localg);
        prb.reset();
        auto start = TIC();
        prb.run(T);
        auto tottime = TOC(start);
        cout << "time(" << i << "): " << tottime << endl;
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, path + name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, path + name+"_pseudoregrets.csv.gz", EVERY, PREC);



    //return 0;
    //LEADER VS MODEL SELECTION BASELINES

    //EXP3.P
    name = "EXP3dotP";
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
                FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);

        ContBanditProblem<int> prb(cpRefRep, localg);
        prb.reset();
        auto start = TIC();
        prb.run(T);
        auto tottime = TOC(start);
        cout << "time(" << i << "): " << tottime << endl;
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, name+"_pseudoregrets.csv.gz", EVERY, PREC);


    //Corral
    name = "Corral";
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
                FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);

        ContBanditProblem<int> prb(cpRefRep, localg);
        prb.reset();
        auto start = TIC();
        prb.run(T);
        auto tottime = TOC(start);
        cout << "time(" << i << "): " << tottime << endl;
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, name+"_pseudoregrets.csv.gz", EVERY, PREC);

    //EXP4.IX
    name = "EXP4dotIX";
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
         double exp4_gamma = sqrt(2*log(base_algs.size())/(reference_rep.n_arms()*T));
         double exp4_lr = 2*exp4_gamma;
         EXP4dotIX<int> localg(base_algs, exp4_lr, exp4_gamma, seeds[i]);
                 FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);

         ContBanditProblem<int> prb(cpRefRep, localg);
         prb.reset();
         auto start = TIC();
         prb.run(T);
         auto tottime = TOC(start);
         cout << "time(" << i << "): " << tottime << endl;
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
    }
    save_vector_csv_gzip(regrets, name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, name+"_pseudoregrets.csv.gz", EVERY, PREC);


    //REGRET Balancing
    name = "OFULBAL";
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
         RegretBalance<int> localg(base_algs);
                 FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);

         ContBanditProblem<int> prb(cpRefRep, localg);
         prb.reset();
         auto start = TIC();
         prb.run(T);
         auto tottime = TOC(start);
         cout << "time(" << i << "): " << tottime << endl;
         regrets[i] = prb.instant_regret;
         pseudo_regrets[i] = prb.exp_instant_regret;
         // delete localg;
    }
    save_vector_csv_gzip(regrets, name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, name+"_pseudoregrets.csv.gz", EVERY, PREC);


    //Regret balancing with elimination
    name = "OFULBALELIM";
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
        RegretBalanceAndEliminate<int> localg(base_algs, delta);
                FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);

        ContBanditProblem<int> prb(cpRefRep, localg);
        prb.reset();
        auto start = TIC();
        prb.run(T);
        auto tottime = TOC(start);
        cout << "time(" << i << "): " << tottime << endl;
        regrets[i] = prb.instant_regret;
        pseudo_regrets[i] = prb.exp_instant_regret;
        // delete localg;
    }
    save_vector_csv_gzip(regrets, name +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, name+"_pseudoregrets.csv.gz", EVERY, PREC);


    return 0;
}
