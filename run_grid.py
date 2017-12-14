#!/usr/bin/python
import sys
sys.path.append("../../")
from nova.gridmaker import get_grid
from nova.io import IO, ImportExport
from nova.optimize import Genetic
from nova.run import MakeRun
import logging
import time
import json
from tqdm import tqdm


def read_grid(filename):
    with open(filename + ".json") as grid:
        return json.load(grid)


def run_main(seed_tables=None, n_proc=2, norm=None, statistic="chisq", con=None, max_iterations=1,
             max_generations=1, obs=None, model_name="", grid=None, emis_file=None, path=None, n_parents=None,
             params_vary=None, params_limit=None, loadmodels=True, generate=False, make=False, iterate=False):
    parent_models = []
    stop_iterations = False
    i_iterations = 0
    while not stop_iterations:
        i_iterations += 1
        stop_generations = False
        # logger.info("Iteration {}".format(i_iterations))
        # print("Iteration {}".format(i_iterations))

        opt = Genetic(engine=con, generations=max_generations)

        if isinstance(params_limit, dict):
            opt.set_param_limits(params_limit)
        elif isinstance(params_limit, tuple):
            opt.set_param_limits(*params_limit)

        # Set normalization
        if isinstance(norm, list):
            do_normalize = True
            for n in norm:
                opt.set_norm(*n)
        else:
            do_normalize = False

        # Set observation
        if obs is not None:
            opt.set_obs(obs, norm=False)
        if do_normalize:
            opt.normalize_all(which="obs")

        if generate:
            count = 0
        else:
            count = 1
        i_gen = 0
        while not stop_generations:
            logger.info("Iteration {}".format(i_iterations))
            print("Iteration {}".format(i_iterations))

            count += 1
            # print(icount)
            # print(grid)

            # Make input files and run Cloudy
            if count > 1:
                if make:
                    # mr = MakeRun(path=path, version=cloudy_version)
                    mr.make_input(model_name=model_name, grid=grid, emis_file=emis_file)
                    mr.run_cloudy(n_proc=n_proc)

                if loadmodels:
                    # Load models and export to database
                    ie.load_models(path=path, read_grains=True)
                    ie.to_sql()

            # Generate new models
            if generate:
                i_gen += 1
                if count == 1:
                    if i_iterations == 1:
                        opt.set_models(seed_tables, norm=False)
                        opt.normalize_all(which="model")
                        # tables_list = seed_tables
                    else:
                        opt.set_models(parent_models, norm=False)
                        opt.normalize_all(which="model")
                        # tables_list = parent_models
                        print("Using parent models as input")
                else:
                    tables_list = ie.format_table(ie.tables_list)  # deepcopy(ie.tables_list)
                    opt.set_models(tables_list, norm=False)
                    opt.normalize_all(which="model")
                # opt.set_models(tables_list, norm=False)

                # Normalize model data
                # opt.normalize_all(which="model")
                # debug = opt._cloudy_models[0]["model_params"].set_index("param").loc["other_opt", "value"]#.values[0]
                # print("-----------------------------------------------")
                # print(debug)
                # print(type(debug))

                # Run optimization
                opt.fit(method=statistic)
                parent_models.extend(opt.parents_final)

                # if opt.stop(i_gen=i_gen):
                #    tmp_id = []
                #    for parent in opt.parents_final:
                #        print(opt.fit_res)
                #        print(parent["model_info"])
                #        if parent["fit_res"] < min(opt.fit_res) and parent["model_info"].id.values[0] not in tmp_id:
                #            tmp_id.append(parent["model_info"].id.values[0])
                #            tables_list.append(parent)
                #    seed_tables = tables_list
                #    stop_generations = True
                if opt.stop(i_gen=i_gen):
                    # print(parent_models)
                    break

                # print("Best fit models: {}".format(opt.best_fit(10)))
                stop = opt.new_generation(params=params_vary, n_parents=n_parents, n_children=n_children,
                                          i_gen=i_gen, p=0.6, dfrac=0.5)  # 0.5)

                grid = get_grid(opt.get_children(), is_generated=True)
                # print(opt.parents_final)
                # parent_models.extend(opt.parents_final)
                # i_gen += 1
            else:
                stop_generations = True

        if i_iterations >= max_iterations:
            stop_iterations = True

    logger.info("Stopping...")
    print("Stopping...")
    return opt

def main(argv):
    path = "./"
    name = ''
    ncpu = 1

    opts, args = getopt.getopt(argv, "hn:ncpu:", ["name=", "ncpu="])
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -n <filename> -ncpu <1>')
            sys.exit()
        elif opt in ("-path", "--model_path"):
            path = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-ncpu"):
            ncpu = arg
    print('Model name is ', name)

    cloudy_version = "c13"
    iterate = False
    make = True
    generate = False
    max_iterations = 0
    max_generations = 0
    n_parents = 1
    n_children = 1

    # Open connection to DB. Lives throughout session.
    #io = IO(db="../db/{}.db".format(name))
    #io.connect_to_db()
    #db_engine = io.engine  # The DB connection

    # Read grid
    grid = read_grid("runs/savegrids/" + name)

    mr = MakeRun(path=path, version=cloudy_version, iterator=None)
    #ie = ImportExport(engine=db_engine)
    n_proc = ncpu

    start_time = time.time()
    opt = run_main(seed_tables=None, n_proc=n_proc, norm=None, con=None, max_iterations=max_iterations,
                   max_generations=max_generations, obs=None, model_name=model_name, emis_file=emis_file, path=path,
                   n_parents=n_parents, params_vary=None, params_limit=None, loadmodels=False, generate=generate,
                   make=make, iterate=iterate, grid=grid)

    logger.info("--- %s minutes ---" % ((time.time() - start_time) / 60))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


if __name__ == "__main__":
    # Create logger
    logger = logging.getLogger('NOVA')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('../runs/{}.log'.format(model_name))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("\n==================================================================\n"
                "\n NEW RUN STARTED"
                "\n==================================================================")