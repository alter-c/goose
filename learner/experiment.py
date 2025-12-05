"""
Main script for running *domain-dependent* GOOSE experiments. The experiment pipeline consists of
1. training 3 models for each (domain, graph_representation, GNN) configuration
2. validating the 3 models with search
3. selecting the best model from the 3 models
4. evaluating the best model on test problems

Example command: 
nohup python -u experiment.py >> goose.log &
nohup python -u experiment.py -r lpig >> experiment.log &
"""

import os
import argparse
import json
import logging

import numpy as np
from itertools import product
from representation import REPRESENTATIONS
from util.scrape_log import scrape_search_log, scrape_train_log, search_finished_correctly
from util.search import ROOT_DIR, FAIL_LIMIT, fd_cmd, sorted_nicely
from dataset.goose_domain_info import GOOSE_DOMAINS
import multiprocessing

VAL_REPEATS = 3
TIMEOUT = 600 + 30 # 600 seconds for search, 30 seconds for warmup

_SEARCH = "gbbfs"

_MODEL_NAME = "attention"
_EXPERIMENT_DIR = os.path.join(ROOT_DIR, "experiment", "results", _MODEL_NAME)

_TRAINED_MODEL_DIR = f"{_EXPERIMENT_DIR}/all_trained_models"
_VALIDATED_MODEL_DIR = f"{_EXPERIMENT_DIR}/trained_models"

_MAIN_LOG_DIR = f"{_EXPERIMENT_DIR}/logs"
_LOG_DIR_TRAIN = f"{_MAIN_LOG_DIR}/train"
_LOG_DIR_VAL = f"{_MAIN_LOG_DIR}/val"
_LOG_DIR_SELECT = f"{_MAIN_LOG_DIR}/select"
_LOG_DIR_TEST = f"{_MAIN_LOG_DIR}/test"

os.makedirs(_TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(_VALIDATED_MODEL_DIR, exist_ok=True)
os.makedirs(_MAIN_LOG_DIR, exist_ok=True)
os.makedirs(_LOG_DIR_TRAIN, exist_ok=True)
os.makedirs(_LOG_DIR_VAL, exist_ok=True)
os.makedirs(_LOG_DIR_SELECT, exist_ok=True)
os.makedirs(_LOG_DIR_TEST, exist_ok=True)

dump_file = f"{_EXPERIMENT_DIR}/coverage.json"
logging.basicConfig(
    filename=f"{_EXPERIMENT_DIR}/exp.log",
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def p_log(log):
    print(log)
    logging.info(log)


def get_model_desc(rep, domain, L, H, aggr, val_repeat, patience, lr, batch_size):
    return f"{_MODEL_NAME}_{rep}_{domain}_L{L}_H{H}_{aggr}_p{patience}_lr{lr}_bs{batch_size}_v{val_repeat}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", choices=GOOSE_DOMAINS, default=None)
    parser.add_argument("-r", "--representation", choices=REPRESENTATIONS, default="rlg")
    args = parser.parse_args()

    # fixed params
    rep = args.representation
    domain = args.domain
    device = 0 
    aggr = "mean"
    seed = 3407
    
    # training params
    lr = 0.001
    batch_size = 32

    # test params
    domains = GOOSE_DOMAINS if args.domain is None else [args.domain]
    Ls = [1]
    Hs = [16]
    patiences = [10]
    
    param_tuple = list(product(domains, Ls, Hs, patiences))

    param_tuple.remove(("blocks", 1, 16, 10))
    param_tuple.append(("blocks", 2, 16, 50))
    param_tuple.remove(("spanner", 1, 16, 10))
    param_tuple.append(("spanner", 2, 8, 10))


    def run_experiment_for_domain(args_tuple):
        domain, L, H, patience = args_tuple
        val_dir = f"../dataset/goose/{domain}/val"
        test_dir = f"../dataset/goose/{domain}/test"
        df = f"../dataset/goose/{domain}/domain.pddl"

        """ train """
        for val_repeat in range(VAL_REPEATS):
            os.system("date")

            desc = get_model_desc(rep, domain, L, H, aggr, val_repeat, patience, lr, batch_size)
            model_file = f"{_TRAINED_MODEL_DIR}/{desc}.dt"

            train_log_file = f"{_LOG_DIR_TRAIN}/{desc}.log"

            if not os.path.exists(model_file) or not os.path.exists(train_log_file):
                cmd = f"python3 train_gnn.py {domain} -r {rep} -L {L} -H {H} --aggr {aggr} --save-file {model_file} --device {device} --seed {seed} --patience {patience}"
                p_log(f"training with {domain} {rep}, see {train_log_file}")
                os.system(f"{cmd} > {train_log_file}")
            else:
                p_log(f"already trained for {domain} {rep}, see {train_log_file}")
        #######################################################################

        """validate"""
        for val_repeat in range(VAL_REPEATS):
            desc = get_model_desc(rep, domain, L, H, aggr, val_repeat, patience, lr, batch_size)
            model_file = f"{_TRAINED_MODEL_DIR}/{desc}.dt"

            for f in os.listdir(val_dir):
                os.system("date")

                val_log_file = f"{_LOG_DIR_VAL}/{f.replace('.pddl', '')}_{desc}.log"

                finished_correctly = False
                if os.path.exists(val_log_file):
                    finished_correctly = search_finished_correctly(val_log_file)
                if not finished_correctly:
                    pf = f"{val_dir}/{f}"
                    cmd, intermediate_file = fd_cmd(df=df, pf=pf, m=model_file, search=_SEARCH, timeout=TIMEOUT)
                    p_log(f"validating {domain} {rep}, see {val_log_file}")
                    os.system(f"{cmd} > {val_log_file}")
                    if os.path.exists(intermediate_file):
                        os.remove(intermediate_file)
                else:
                    p_log(f"already validated for {domain} {rep}, see {val_log_file}")
        #######################################################################

        """ selection """
        # after running all validation repeats, we pick the best one
        best_model = -1
        best_solved = 0
        best_expansions = float("inf")
        best_runtimes = float("inf")
        best_loss = float("inf")
        best_train_time = float("inf")

        # see if any model solved anything
        for val_repeat in range(VAL_REPEATS):
            desc = get_model_desc(rep, domain, L, H, aggr, val_repeat, patience, lr, batch_size)

            solved = 0
            for f in os.listdir(val_dir):
                val_log_file = f"{_LOG_DIR_VAL}/{f.replace('.pddl', '')}_{desc}.log"
                stats = scrape_search_log(val_log_file)
                solved += stats["solved"]
            best_solved = max(best_solved, solved)

        # break ties
        for val_repeat in range(VAL_REPEATS):
            desc = get_model_desc(rep, domain, L, H, aggr, val_repeat, patience, lr, batch_size)
            model_file = f"{_TRAINED_MODEL_DIR}/{desc}.dt"

            solved = 0
            expansions = []
            runtimes = []
            for f in os.listdir(val_dir):
                val_log_file = f"{_LOG_DIR_VAL}/{f.replace('.pddl', '')}_{desc}.log"
                stats = scrape_search_log(val_log_file)
                solved += stats["solved"]
                if stats["solved"]:
                    expansions.append(stats["expanded"])
                    runtimes.append(stats["time"])
            expansions = np.median(expansions) if len(expansions) > 0 else -1
            runtimes = np.median(runtimes) if len(runtimes) > 0 else -1
            train_stats = scrape_train_log(f"{_LOG_DIR_TRAIN}/{desc}.log")
            avg_loss = train_stats["best_avg_loss"]
            train_time = train_stats["time"]
            # choose best model
            if (solved == best_solved and best_solved > 0 and expansions < best_expansions) or (
                solved == best_solved and best_solved == 0 and avg_loss < best_loss
            ):
                best_model = model_file
                best_expansions = expansions
                best_runtimes = runtimes
                best_loss = avg_loss
                best_train_time = train_time

        # log best model stats
        desc = f"{_MODEL_NAME}_{rep}_{domain}_L{L}_H{H}_{aggr}_p{patience}_lr{lr}_bs{batch_size}"
        best_model_file = f"{_VALIDATED_MODEL_DIR}/{desc}.dt"
        with open(f"{_LOG_DIR_SELECT}/{desc}.log", "w") as f:
            f.write(f"model: {best_model}\n")
            f.write(f"solved: {best_solved} / {len(os.listdir(val_dir))}\n")
            f.write(f"median_expansions: {best_expansions}\n")
            f.write(f"median_runtime: {best_runtimes}\n")
            f.write(f"avg_loss: {best_loss}\n")
            f.write(f"train_time: {best_train_time}\n")
            f.close()
        os.system(f"cp {best_model} {best_model_file}")
        #######################################################################

        """ test """
        success = 0
        failed = 0

        # warmup first
        f = sorted_nicely(os.listdir(test_dir))[0]
        pf = f"{test_dir}/{f}"
        cmd, intermediate_file = fd_cmd(df=df, pf=pf, m=model_file, search=_SEARCH, timeout=30)
        os.system("date")
        p_log(f"warming up with {domain} {rep} {f.replace('.pddl', '')} {best_model_file}")
        os.popen(cmd).readlines()
        try:
            os.remove(intermediate_file)
        except OSError:
            pass

        # test on problems(for whole domain)
        test_list = sorted_nicely(os.listdir(test_dir))
        for f in test_list:
            os.system("date")
            test_log_file = f"{_LOG_DIR_TEST}/{f.replace('.pddl', '')}_{desc}.log"
            finished_correctly = False
            if os.path.exists(test_log_file):
                finished_correctly = search_finished_correctly(test_log_file)
            if not finished_correctly:
                pf = f"{test_dir}/{f}"
                cmd, intermediate_file = fd_cmd(df=df, pf=pf, m=model_file, search=_SEARCH, timeout=TIMEOUT)
                p_log(f"testing {domain} {rep}, see {test_log_file}")
                os.system(f"{cmd} > {test_log_file}")
                if os.path.exists(intermediate_file):
                    os.remove(intermediate_file)
            else:
                p_log(f"already tested for {domain} {rep}, see {test_log_file}")

            # check if failed or not
            assert os.path.exists(test_log_file)
            log = open(test_log_file, "r").read()
            solved = "Solution found." in log
            if solved:
                failed = 0
                success += 1
                log = f"solved: {success} / {len(test_list)}"
                print(log, flush=True)
                logging.info(log)
            else:
                failed += 1
                print("failed", flush=True)
                logging.warning("failed")
            if failed >= FAIL_LIMIT[domain]:
                print("Stop testing due to continuous fails.")
                logging.error("Stop testing due to continuous fails.")
                break

        coverage_ratio = success / len(test_list)
        p_log(f"Coverage ratio of {domain} is {coverage_ratio:.4f}. (Params: L={L}, H={H}, p={patience})")

        # write to json
        result_obj = {
            "model name": _MODEL_NAME,
            "domain": domain,
            "representation": rep,
            "network params":{
                "RGNN layers": L,
                "hidden dimension": H,
                "aggregator": aggr
            },
            "training params": {
                "patience": patience,
                "learning rate": lr,
                "batch size": batch_size
            },
            "seed": seed if 'seed' in locals() else 3407,
            "coverage": coverage_ratio
        }

        if os.path.exists(dump_file) and os.path.getsize(dump_file) > 0:
            with open(dump_file, "r") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(result_obj)
        with open(dump_file, "w") as f:
            json.dump(data, f)

        return coverage_ratio

    workers = 1

    # Use multiprocessing to run experiments in parallel
    with multiprocessing.Pool(processes=min(workers, len(param_tuple), os.cpu_count())) as pool:
        pool.map(run_experiment_for_domain, param_tuple)


