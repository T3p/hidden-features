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

    int REF_REPR = 0;
    std::vector<std::string> files{
        "../../problem_data/lastfm/repr/lastfm_post_13_d10_span10_gt.npz",
        "../../problem_data/lastfm/repr/lastfm_post_0_d43_span43.npz",
        "../../problem_data/lastfm/repr/lastfm_post_1_d44_span43.npz",
        "../../problem_data/lastfm/repr/lastfm_post_2_d20_span19.npz",
        "../../problem_data/lastfm/repr/lastfm_post_3_d28_span27.npz",
        "../../problem_data/lastfm/repr/lastfm_post_4_d39_span39.npz",
        "../../problem_data/lastfm/repr/lastfm_post_5_d13_span13.npz",
        "../../problem_data/lastfm/repr/lastfm_post_6_d23_span21.npz",
        "../../problem_data/lastfm/repr/lastfm_post_7_d31_span31.npz",
        "../../problem_data/lastfm/repr/lastfm_post_8_d24_span24.npz",
        "../../problem_data/lastfm/repr/lastfm_post_9_d20_span20.npz",
        "../../problem_data/lastfm/repr/lastfm_post_10_d44_span44.npz",
        "../../problem_data/lastfm/repr/lastfm_post_11_d9_span8.npz",
        "../../problem_data/lastfm/repr/lastfm_post_12_d28_span27.npz",
        "../../problem_data/lastfm/repr/lastfm_post_14_d40_span40.npz",
        "../../problem_data/lastfm/repr/lastfm_post_15_d7_span6.npz",
        "../../problem_data/lastfm/repr/lastfm_post_16_d14_span14.npz",
        "../../problem_data/lastfm/repr/lastfm_post_17_d15_span15.npz",
        "../../problem_data/lastfm/repr/lastfm_post_18_d37_span35.npz",
        "../../problem_data/lastfm/repr/lastfm_post_19_d37_span37.npz"
    };
    std::vector<std::string> names{
        "0_d43_span43",
        "1_d44_span43",
        "2_d20_span19",
        "3_d28_span27",
        "4_d39_span39",
        "5_d13_span13",
        "6_d23_span21",
        "7_d31_span31",
        "8_d24_span24",
        "9_d20_span20",
        "10_d44_span44",
        "11_d9_span8",
        "12_d28_span27",
        "13_d10_span10_gt",
        "14_d40_span40",
        "15_d7_span6",
        "16_d14_span14",
        "17_d15_span15",
        "18_d37_span35",
        "19_d37_span37",
    };

    std::time_t t = std::time(nullptr);
    char MY_TIME[100];
    std::strftime(MY_TIME, sizeof(MY_TIME), "%Y%m%d%H%M%S", std::localtime(&t));
    std::cout << MY_TIME << '\n';

    typedef std::vector<std::vector<double>> vec2double;

    int seed = time(NULL);
    srand (seed);
    cout << "seed: " << seed << endl;
    int n_runs = 1, T = 100000;
    double delta = 0.01;
    double reg_val = 1.;
    double noise_std = 0.3;
    double bonus_scale = 1.;
    bool adaptive_ci = true;

    std::vector<long> seeds(n_runs);
    std::generate(seeds.begin(), seeds.end(), [] ()
    {
        return rand();
    });

    // load reference representation
    FiniteLinearRepresentation reference_rep=flr_loadnpz(files[REF_REPR], noise_std, seed, "features", "theta");
    cout << "Ref_rep.dim: " << reference_rep.features_dim() << endl;
    cout << "Ref_rep.feat_bound=" << reference_rep.features_bound() << endl;

    // other representations
    std::vector<FiniteLinearRepresentation> reps;
    for(int i =0; i < files.size(); ++i) {
        FiniteLinearRepresentation rr = flr_loadnpz(files[i], noise_std, seed, "features", "theta");
        cout << "phi_" << i << ".dim=" << rr.features_dim() << endl;
        cout << "phi_" << i << ".feat_bound=" << rr.features_bound() << endl;
        reps.push_back(rr);
    }

    vec2double regrets, pseudo_regrets;
    #pragma omp parallel for
    for (int i = 0; i < n_runs; ++i)
    {
        std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
        for(auto& ll : reps)
        {
            auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
            lreps.push_back(tmp);
        }
        LEADER_MSEELIM<int> localg(lreps, reg_val, noise_std, bonus_scale, delta/lreps.size(), adaptive_ci);

        // create same representation but witth different seed
        FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);
        ContBanditProblem<int> prb(cpRefRep, localg);
        prb.reset();
        auto start = TIC();
        prb.run(T);
        auto tottime = TOC(start);
        cout << "time(" << i << "): " << tottime << endl;
        regrets.push_back(prb.instant_regret);
        pseudo_regrets.push_back(prb.exp_instant_regret);

        save_vector_csv_gzip(regrets, "LEADER-"+std::string(MY_TIME)+"_regrets.csv.gz", EVERY, PREC);
        save_vector_csv_gzip(pseudo_regrets, "LEADER-"+std::string(MY_TIME)+"_pseudoregrets.csv.gz", EVERY, PREC);
    }

    // //just OFUL
    // for(int j = 0; j < reps.size(); ++j)
    // {
    //     vec2double regrets(n_runs), pseudo_regrets(n_runs);

    //     #pragma omp parallel for
    //     for (int i = 0; i < n_runs; ++i)
    //     {
    //         FiniteLinearRepresentation lrep = reps[j].copy(seeds[i]);
    //         OFUL<int> localg(lrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
    //         // create same representation but witth different seed
    //         FiniteLinearRepresentation cpRefRep = reference_rep.copy(seeds[i]);
    //         ContBanditProblem<int> prb(cpRefRep, localg);
    //         prb.reset();
    //         auto start = TIC();
    //         prb.run(T);
    //         auto tottime = TOC(start);
    //         regrets[i] = prb.instant_regret;
    //         pseudo_regrets[i] = prb.exp_instant_regret;
    //     }
    //     // save_vector_csv(regrets, "OFUL-rep"+std::to_string(j)+"_regrets.csv", EVERY, PREC);
    //     save_vector_csv_gzip(regrets, "OFUL-rep"+std::to_string(j)+"_regrets.csv.gz", EVERY, PREC);
    //     // save_vector_csv(pseudo_regrets, "OFUL-rep"+std::to_string(j)+"_pseudoregrets.csv", EVERY, PREC);
    //     save_vector_csv_gzip(pseudo_regrets, "OFUL-rep"+std::to_string(j)+"_pseudoregrets.csv.gz", EVERY, PREC);
    // }


    //Regret balancing with elimination
    bool update_all = true;
    std::string name = "OFULBALELIM";
    std::cout << name << std::endl;
    regrets.resize(n_runs);
    pseudo_regrets.resize(n_runs);
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
        // RegretBalanceAndEliminate<int> localg(base_algs, delta, update_all);
        RegretBalance<int> localg(base_algs, update_all);
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
    save_vector_csv_gzip(regrets, name+std::string(MY_TIME) +"_regrets.csv.gz", EVERY, PREC);
    save_vector_csv_gzip(pseudo_regrets, name+std::string(MY_TIME)+"_pseudoregrets.csv.gz", EVERY, PREC);

    return 0;
}
