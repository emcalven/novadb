import numpy as np
from collections import OrderedDict
import pandas as pd
import logging
import os
from tqdm import tqdm
import time

# create logger
module_logger = logging.getLogger('NOVA.optimize')


class Optimize(object):

    def __init__(self, engine=None):
        """
        :param db: Database connection object
        :param obs: Observed spectrum: pd.Dataframe
        :param obs_err: Error in observed spectrum: pd.DataFrame
        :param cloudy_models: List of dictionaries: Expects format:
                                  [{'model_info':pd.DataFrame, 'lines':pd.DataFrame,
                                  'ionic_frac':pd.DataFrame, 'abund':pd.DataFrame, 'model_input':pd.DataFrame}, ...]
        :param norm: Normalizing line: str
        """

        # self.logger = logging.getLogger('NOVA.optimize.Optimize')
        # self.logger.info('creating an instance of Optimize')

        self._engine = engine
        self._obs = None
        self._obs_err = None
        self._cloudy_models = None
        self._id_table = None
        self._lines_table_models = None
        self._dropped_lines = None
        self._norm = Norm()  # None
        self._seed_params = None
        self.fit_res = None
        self.model_final = None
        self.obs_final = None
        self.obs_err_final = None

    def set_engine(self, engine):
        self._engine = engine

    def set_obs(self, obs, obs_err=None, norm=True):
        self._obs = obs
        if self._obs.index.name is not "line_id":
            self._obs.set_index("line_id", inplace=True)
        self._obs.sort_index(inplace=True)
        # TODO: Properly normalize errors!
        if obs_err is not None:
            self._obs["err"] = obs_err
        if norm:
            self._obs = self.normalize(self._obs)

        #if self._obs_err is not None:
        #    if self._obs_err.index.name is not "line_id":
        #        self._obs_err.set_index("line_id", inplace=True)
            # Normalize observed flux error
        #    self._obs_err = self.normalize(self._obs, err=True)
        # Normalize observed flux
        #    self._obs = self.normalize(self._obs)

    def set_obs_err(self, obs_err):
        # TODO: Apply normalization on obs errors
        self._obs_err = obs_err
        if self._obs_err.index.name is not "line_id":
            self._obs_err.set_index("line_id", inplace=True)

    def add_model(self, model, norm=False):
        self._cloudy_models.append(model)
        model_lines = model["lines"].set_index("line_id", inplace=False)
        if "model_id" in model_lines:
            model_lines = model_lines.drop("model_id", axis=1)
        if norm:
            model_lines = self.normalize(model_lines)
        self._lines_table_models.append(model_lines)
        try:
            if model["model_info"].index.name == "id":
                self._id_table.append(model["model_info"].index.tolist())
            else:
                self._id_table.append(model["model_info"]["id"].values[0])
        except:
            print("model_info does not have id in index or field.")

    #def set_start_params(selfs, pardict):
    #    """Modify existing model parameter values"""
    #    df_out = df.copy()
    #    df_out.reset_index(inplace=True)
    #    df_out = df_out.set_index("param", drop=False)
    #    for key, val in pardict.items():
    #        df_out.loc[key, "value"] = val
    #    return df_out.set_index("model_id", drop=True)

    def set_models(self, models, norm=False, json=False, seed_params=None):
        self._cloudy_models = []
        self._seed_params = seed_params  # Parameters to change
        #self._cloudy_models = models
        #if mixing:
        #    self._cloudy_models = {"mixed": models}
        #else:
        #    if self._cloudy_models is None:
        #        self._cloudy_models = {Tree._id(m):m for m in models}
        #    else:
        #        self._cloudy_models.update({Genetic.model_progenitor(m):m for m in models})

        #if not isinstance(self._cloudy_models, list):
        if not isinstance(models, list):
            #self._cloudy_models = [self._cloudy_models]
            models = [models]

        self._lines_table_models = []
        self._id_table = []
        #for model in self._cloudy_models:
        for model in models:
            self.add_model(model, norm=norm)

        #    model_lines = model["lines"].set_index("line_id", inplace=False)
        #    if "model_id" in model_lines:
        #        model_lines = model_lines.drop("model_id", axis=1)
        #    if norm:
        #        model_lines = self.normalize(model_lines)
        #    self._lines_table_models.append(model_lines)
        #    #self._id_table.append(model["model_info"]["id"].values[0])
        #    try:
        #        if model["model_info"].index.name == "id":
        #            self._id_table.append(model["model_info"].index.tolist())
        #        else:
        #            self._id_table.append(model["model_info"]["id"].values[0])
        #    except:
        #        print("model_info does not have id in index or field.")

    def set_norm(self, norm=None, lines=None, norm_dict=None):
        #self._norm = norm
        self._norm.add_norm(norm=norm, lines=lines, norm_dict=norm_dict)

    def normalize_all(self, which="all"):
        if which == "all" or which == "model":
            # Normalize model lines
            self._lines_table_models = []
            print("Normalizing models...")
            time.sleep(0.3)
            for model in tqdm(self._cloudy_models):
                model_lines = model["lines"].set_index("line_id", inplace=False)
                model_lines = self.normalize(model_lines)
                if model_lines.empty:
                    print("Failed normalizing model")
                else:
                    self._lines_table_models.append(model_lines)

        if which == "all" or which == "obs":
            # Normalize observed lines
            obs = self.normalize(self._obs)
            if obs.empty:
                print("Failed normalizing observation")
                raise Exception('Failed normalizing observation. Aborting run...')
            else:
                self._obs = obs

    def normalize(self, data):
        norm = self._norm.norm.copy()
        # Check if all normalizing lines exists in data, return False if not
        differing_lines = list(set(list(norm.keys())).difference(set(list(data.index))))
        if differing_lines:
            print("Normalizing lines {} not in data. Aborting...".format(differing_lines))
            return pd.DataFrame()
        # Drop line to be normalized if not existing in data
        for key, val in norm.items():
            differing_lines = list(set(list(val)).difference(set(list(data.index))))
            matching_lines = list(set(list(val)).intersection(set(list(data.index))))
            #print(data)
            #data = data.loc(matching_lines)
            norm[key] = matching_lines
            #print("Dropping lines {} not existing in data...".format(differing_lines))
            #if differing_lines:
            #    print("Dropping lines {} from norm...".format(differing_lines))
        normalized_data = self._norm.normalize(data, norm=norm)
        # TODO: Handle case when no normalized data is return (None)
        return normalized_data

    def _conform_lines(self, model):
        """Method to select from models only those lines that appear in
           observation.

           :Return
        """
        obs_tmp = self._obs.copy()
        #obs_err_tmp = self._obs_err.copy()
        if "model_id" in model:
            model = model.drop("model_id", axis=1)
        line_difference = list(set(list(model.index)).difference(set(list(self._obs.index))))
        model = model.drop(line_difference)
        if self._dropped_lines is None:
            self._dropped_lines = [list(set(list(self._obs.index)).difference(set(list(model.index))))]
        else:
            self._dropped_lines.append(list(set(list(self._obs.index)).difference(set(list(model.index)))))
        if len(obs_tmp.index) > len(model.index):
            line_difference = list(set(list(obs_tmp.index)).difference(set(list(model.index))))
            obs_tmp = obs_tmp.drop(line_difference)
            #obs_err_tmp = obs_err_tmp.drop(line_difference)
        #print(model)
        #print(obs_tmp)
        #print(obs_err_tmp)
        return model, obs_tmp#, obs_err_tmp

    def _sanity_check_pass(self, model, obs):
        if list(obs.index) == list(model.index):  # == list(obs_err.index):
            return True
        else:
            return False

    @staticmethod
    def chisq(m, o, oerr, log=False):
        """"m, o, oerr expected to be numpy arrays"""
        if log:
            return np.sum((np.log10(m) - np.log10(o)) ** 2 / np.log10(oerr) ** 2)
        else:
            return np.sum((m - o) ** 2 / oerr ** 2)

    @staticmethod
    def mse(m, o, log=False):
        """m and o expected to be numpy arrays"""
        n = len(m)
        if log:
            return (1. / n) * np.sum((np.log10(m) - np.log10(o)) ** 2)
        else:
            return (1. / n) * np.sum((m - o) ** 2)

    def fit(self, n=1, method="mse", log=False):
        """Do the fitting"""
        if self._obs is None:
            return None
        if "err" not in self._obs.columns:
            self._obs["err"] = self._obs.loc[:, "value"] * 0.5
            print("Using artifical errors...")
            #self._obs_err.loc[:, "value"] *= 0.1
        #if self._obs_err is None:
        #    self._obs_err = self._obs.copy()
        #    self._obs_err.loc[:, "value"] *= 0.1
        # if not isinstance(self._cloudy_models, list):
        #     self._cloudy_models = [self._cloudy_models] Duplicate check of this criteria

        fit_res = []
        for id_, ml in zip(self._id_table, self._lines_table_models):
            ml, obs = self._conform_lines(ml)
            ml = ml.sort_index()
            obs = obs.loc[ml.index]
            # obs_err = obs_err.loc[ml.index]
            # obs.sort_index(inplace=True)
            # obs_err.sort_index(inplace=True)
            if self._sanity_check_pass(ml, obs):
                if method == "chisq":
                    # fit_res.append(self.chisq(ml, self._obs, self._obs_err, log=log).values[0])
                    # print(ml)
                    # print(obs)
                    # print(obs_err)
                    fit_res.append(self.chisq(ml.value, obs.value, obs.err, log=log))  # .values[0])
                elif method == "mse":
                    # fit_res.append(self.mse(ml, self._obs, log=log).values[0])
                    # print(ml.value)
                    # print(obs.value)
                    fit_res.append(self.mse(ml.value, obs.value, log=log))#.values[0])
        self.fit_res = np.array(fit_res)

    def best_fit(self, n=1):
        if self.fit_res is not None:
            self.model_final = []
            self.obs_final = []
            self.obs_err_final = []
            best_models_idx = self.fit_res.argsort()[:n]
            print(best_models_idx)
            for idx in best_models_idx:
                m, obs = self._conform_lines(self._lines_table_models[idx])
                obs = obs.sort_values(by="wavelength", ascending=True)
                self.obs_final.append(obs)
                self.model_final.append(m.loc[obs.index])
                # self.obs_err_final.append(obs_err.loc[obs.index])
            return best_models_idx #self.fit_res.argsort()[:n]
            # return np.argpartition(self.res, n)[:n]

class Tree(object):

    # __generation__ = 0
    __progenitor__ = None
    __params__ = None

    def __init__(self, ref_abundances=None, param_limits={}):

        self.logger = logging.getLogger('NOVA.optimize.Tree')
        self.logger.info('creating an instance of Tree')
        self._ref_abundances = ref_abundances
        self.parent = None
        self.children = None
        self.parent_params = None
        self.param_limits = param_limits

    @staticmethod
    def progenitor():
        return Tree.__progenitor__

    @staticmethod
    def _id(model):
        try:
            if model["model_info"].index.name == "id":
                #print(model["model_info"].index)
                #print(model["model_info"])
                return model["model_info"].index.tolist()[0]
            else:
                return model["model_info"].id.values[0]
        except:
            print("model_info does not have id in index or field.")

    @staticmethod
    def get_params(model):
        params = model["model_params"].set_index("param", inplace=False)
        if "model_id" in params:
            params = params.drop("model_id", axis=1)
        return params

    def _previous_parent_better(self, parent, change_params=False):
        if self.parent is None:
            return False
        elif change_params is True:
            if 30 * self.parent.get("fit_res") < parent.get("fit_res"):
                print("A")
                return True
            else:
                print("B")
                return False
        elif change_params is False:
            if self.parent.get("fit_res") < parent.get("fit_res"):
                print("C")
                return True
            else:
                print("D")
                return False
        else:
            print("E")
            return False

    def set_parent(self, parent, progenitor=False, change_params=False):
        if progenitor:
            Tree.__progenitor__ = self._id(parent)
            Tree.__params__ = self.get_params(parent)
        is_previous_parent_better = self._previous_parent_better(parent, change_params=change_params)
        #if self._previous_parent_better(parent, changed_params=changed_params):
        print("IS THE PREVIOUS PARENT BETTER", is_previous_parent_better)
        if is_previous_parent_better:
            self.logger.info("Last generation's parent is better. Keeping {} as parent..."
                             "".format(self._id(self.parent)))
            return False
        else:
            self.parent = parent
            self.parent_params = self.get_params(parent)
            print('Parent {} assigned to Tree {}'
                  '\nParent parameters:'
                  '\n{}'.format(self._id(parent), self, self.parent_params.to_dict()["value"]))
            self.logger.info('Parent {} assigned to Tree {}'
                             '\nParent parameters:'
                             '\n{}'.format(self._id(parent), self, self.parent_params.to_dict()["value"]))
            return True

    def set_pars(self, row, params=None, val=None, ref_abundances=None):
        # TODO: Make sure to only enable numerical parameter values to vary
        # print(ref_abundances)
        if ref_abundances is None:
            raise ("Could not find the abundance reference file requested.")

        for par in params:
            if row.name == par:
                # Special case to treat abundances
                if par in ref_abundances:
                    abund_scale = pd.to_numeric(Tree.__params__.loc[par].value)
                    progenitor_abund = np.log10(10 ** ref_abundances.get(par) * abund_scale)
                    delta = progenitor_abund * val
                    # print("Abundance delta {}".format(par))
                    # print(delta)
                    delta_min = np.log10(10 ** progenitor_abund * pd.to_numeric(row.value)) - delta
                    delta_max = np.log10(10 ** progenitor_abund * pd.to_numeric(row.value)) + delta
                    # print("Abundance min, max")
                    # print(delta_min, delta_max)
                    ab = np.random.uniform(delta_min, delta_max, 1)[0]
                    row.value = (10 ** ab) / (10 ** progenitor_abund)
                else:
                    delta = pd.to_numeric(Tree.__params__.loc[par].value) * val
                    delta_min = pd.to_numeric(row.value) - delta
                    delta_max = pd.to_numeric(row.value) + delta
                    row.value = np.random.uniform(delta_min, delta_max, 1)[0]

                    # delta_min, delta_max = self.set_par_limit(par, delta, delta_min, delta_max)
                    row.value = self.new_par_val(par, delta, delta_min, delta_max)

                    # if par == "nh":
                    #     lower_lim = 8e3
                    #     if delta_min <= lower_lim:
                    #         delta_min = lower_lim
                    #         delta_max = delta_min + 2 * delta
                    #     row.value = np.random.uniform(delta_min, delta_max, 1)[0]
                    # if par == "teff":
                    #     lower_lim = 1e5
                    #     if delta_min <= lower_lim:
                    #         delta_min = lower_lim
                    #         delta_max = delta_min + 2 * delta
                    #     row.value = np.random.uniform(delta_min, delta_max, 1)[0]
        return row

    def new_par_val(self, param, delta, delta_min, delta_max):
        if isinstance(self.param_limits.get(param), tuple):
            _min, _max = self.param_limits.get(param)
            if _min is not None:
                if delta_min <= _min:
                    delta_min = _min
                    delta_max = delta_min + 2 * delta
            if _max is not None:
                if delta_max >= _max:
                    delta_max = _max
                    delta_min_tmp = delta_max - 2 * delta
                    if delta_min_tmp >= delta_min:
                        delta_min = delta_min_tmp
            if delta_min >= delta_max:
                delta_min = delta_max
        return np.random.uniform(delta_min, delta_max, 1)[0]


    def generate_child(self, params, i_gen=None, p=None, dfrac=None, seed_params=None, ref_abundances=None):
        """
        Adds a child to self.children with parameters 'pars' randomly drawn from a uniform distribution around
        the ancestors parameter values.

        :param pars: list of parameters
        :param val: amount with which to change the parameter value (d * p**j)
        """
        ref_abundances = self._ref_abundances.get(Tree.__params__.loc["abund_ref"].value)
        val = dfrac * p ** i_gen

        # def set_pars(row, params=None, val=None):
        #     # TODO: Make sure to only enable numerical parameter values to vary
        #     #print(ref_abundances)
        #     if ref_abundances is None:
        #         raise("Could not find the abundance reference file requested.")
        #
        #     for par in params:
        #         if row.name == par:
        #             # Special case to treat abundances
        #             if par in ref_abundances:
        #                 abund_scale = pd.to_numeric(Tree.__params__.loc[par].value)
        #                 progenitor_abund = np.log10(ref_abundances.get(par) * abund_scale)
        #                 delta = progenitor_abund * val
        #                 #(progenitor_abund, delta, pd.to_numeric(row.value))
        #                 delta_min = np.log10(10**progenitor_abund * pd.to_numeric(row.value)) - delta
        #                 delta_max = np.log10(10**progenitor_abund * pd.to_numeric(row.value)) + delta
        #                 #print(delta)
        #                 #print(delta_min, delta_max)
        #                 ab = np.random.uniform(delta_min, delta_max, 1)[0]
        #                 #print(ab, progenitor_abund)
        #                 #print(ab, progenitor_abund)
        #                 #print(val, (10**ab)/(10**progenitor_abund))
        #                 #print("---------------------------------")
        #                 row.value = (10**ab)/(10**progenitor_abund)
        #             else:
        #                 delta = pd.to_numeric(Tree.__params__.loc[par].value) * val
        #                 delta_min = pd.to_numeric(row.value) - delta
        #                 delta_max = pd.to_numeric(row.value) + delta
        #                 row.value = np.random.uniform(delta_min, delta_max, 1)[0]
        #                 if par == "nh":
        #                     lower_lim = 8e3
        #                     if delta_min < lower_lim:
        #                         delta_min = lower_lim
        #                         delta_max = delta_min + 2*delta
        #                     row.value = np.random.uniform(delta_min, delta_max, 1)[0]
        #                 if par == "teff":
        #                     lower_lim = 1e5
        #                     if delta_min < lower_lim:
        #                         delta_min = lower_lim
        #                         delta_max = delta_min + 2*delta
        #                     row.value = np.random.uniform(delta_min, delta_max, 1)[0]
        #                 #print(par)
        #                 #print(delta)
        #                 #print(delta_min, delta_max)
        #                 #print(row.value)
        #                 #print("----------------------------------------------------")
        #
        #     return row

        parent_params = self.parent_params.copy()
        #print("1-----------------------------------------------")
        #print(parent_params.loc["other_opt"])
        #self.logger.info('Raw parent params:\n{}'.format(parent_params))
        parent_params.loc[:, "value"] = parent_params.value.map(lambda x: pd.to_numeric(x, errors="ignore"))  # (A.value, errors="ignore")
        #self.logger.info('After numerization parent params:\n{}'.format(parent_params))
        #print("2-----------------------------------------------")
        #print(parent_params.loc["other_opt"])

        nans = parent_params.loc[~(parent_params.value == parent_params.value)].copy()
        nans.loc[nans.index, "value"] = None
        parent_params.loc[nans.index] = nans

        #ndarrs = parent_params.loc[~(parent_params.value == parent_params.value)].copy()

        #if np.isnan(parent_params.loc["nh_power"].value):
        #    parent_params.loc["nh_power"].value = None
        #    print(None)
        child = parent_params.apply(self.set_pars, params=params, val=val, axis=1, ref_abundances=ref_abundances).to_dict()["value"]
        #print("printing the child:")
        #print(child)
        #child = self.parent_params.apply(set_pars, pars=pars, val=val, axis=1).to_dict()["value"]
        #child["progenitor"] = self.progenitor()
        child = self.add_fields_to_child(child, i_gen, seed_params=seed_params)
        #print(child)
        self.logger.info('Child generated with parameters:\n{}'.format(child))
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

    def add_fields_to_child(self, d, i_gen=0, seed_params=None):
        # Variable
        d["progenitor"] = self.progenitor()
        #if i_gen == 1:
        print(seed_params)
        if seed_params is not None:
            for par, val in seed_params.items():
                d[par] = val
        # Static
        #d["extras"] = OrderedDict(user=self.parent["model_info"].user.values[0],
        #                          project=self.parent["model_info"].project.values[0])
        #d["user"] = self.parent["model_info"].user.values[0]
        #d["project"] = self.parent["model_info"].project.values[0]
        return d


class Genetic(Optimize):

    __param_limits__ = {"nh": (8e3, 1e10),
                        "teff": (1e4, 1e7),
                        "ff": (0, 1),
                        "cf": (0, 1)}
    __params__ = []

    def __init__(self, generations=3, engine=None, ref_abundances=None, param_limits={}):

        """
        Constructs new Cloudy models using a genetic algorithm.

        :param generations:
        :param engine:
        """

        self.logger = logging.getLogger('NOVA.optimize.Genetic')
        self.logger.info('creating an instance of Genetic')
        self.parents_final = []
        self.generations = generations
        self._i_gen = 0
        self._ref_abundances = self._load_ref_abundances(path=ref_abundances)
        self._parent_was_assigned = False
        self.param_limits = Genetic.__param_limits__
        self.params = Genetic.__params__
        super().__init__(engine=engine)

    @staticmethod
    def model_progenitor(model):
        return model["model_info"].progenitor.values[0]

    @staticmethod
    def _generate_children(tree, params=None, n=1, i_gen=0, p=0.5, dfrac=0.1, seed_params=None):
        if params is None:
            raise Exception("Which parameters to vary?")
        else:
            tree.children = None  # Reset children array
            for _ in range(n):
                tree.generate_child(params, i_gen=i_gen, p=p, dfrac=dfrac, seed_params=seed_params)
    @staticmethod
    def _load_ref_abundances(path=None):
        """
        Loads abundance reference json file
        :param path: Path to abundance reference file
        :return: Imported json file in the form of a dictionary
        """
        import json
        if path is None:
            path = 'c13'
            print("Defaulting to reference abundances '{}'".format(path))
        if path == "c13":
            full_path = os.path.dirname(os.path.realpath(__file__)) + "/data/c13.05_abundances.json"
        elif path == "c17":
            full_path = os.path.dirname(os.path.realpath(__file__)) + "/data/c17.00_abundances.json"
        else:
            full_path = path
        try:
            with open(full_path) as ref_abundance_file:
                return json.load(ref_abundance_file)
        except FileNotFoundError:
            print("Could not find the abundance reference file requested [{}]".format(full_path))
            return None

    def set_param_limits(self, p, min=None, max=None):
        if isinstance(p, dict):
            for key, val in p.items():
                if isinstance(val, tuple):
                    if len(val) == 2:
                        self.param_limits[key] = val
                        self.__params__.append(key)
                    else:
                        print("Limit can only be given as (min_val, max_val)")
                else:
                    print("Limit must be a tuple: (min_val, max_val)")
        elif isinstance(p, list):
            for key in p:
                self.param_limits[key] = (None, None)
                self.__params__.append(key)
        else:
            self.param_limits[p] = (min, max)
            self.__params__.append(p)

    def new_generation(self, params=None, n_parents=1, n_children=1, i_gen=0, p=0.5, dfrac=0.1):
        parent_idx = self.best_fit(n=n_parents)  # Get indices of n best models to use as parents
        self.parents_final = []
        self.parents = []
        self.children = []
        self.logger.info('Generation {}/{}'.format(i_gen, self.generations))
        if i_gen == 1:
            self.trees = {}
        for i, idx in enumerate(parent_idx):
            self.logger.info('Parent {}/{}'.format(i+1, len(parent_idx)))
            parent = self._cloudy_models[idx]
            #print(parent["model_params"].set_index("param", inplace=False))
            parent["fit_res"] = self.fit_res[idx]  # Make new field in parent dict containing its fit result
            self.logger.info('Parent {} info: ID = {}, fit result {}'.format(i+1, Tree._id(parent), parent["fit_res"]))
            parent_pars = Tree.get_params(parent).to_dict()["value"]  # Can safely be removed
            parent_pars["progenitor"] = parent["model_info"].progenitor.values[0]  # Can safely be removed
            self.parents.append(parent_pars)  # Can safely be removed
            if i_gen == 1:
                self.logger.info('Creating new Tree...')
                tree = Tree(ref_abundances=self._ref_abundances, param_limits=self.param_limits)
                self.logger.info('Tree created')
                if tree.set_parent(parent, progenitor=True):
                    self.logger.info("Setting the parent...")
                    self.parents_final.append(parent)
                self.logger.info('Parent {} assigned to Tree'.format(Tree._id(parent)))
                self.logger.info('Tree has progenitor {}'.format(tree.progenitor()))
                self.logger.info('Generating children...')
                self._generate_children(tree, params=params, n=n_children, i_gen=i_gen, p=p, dfrac=dfrac,
                                        seed_params=self._seed_params)
                self.trees[tree.progenitor()] = tree
                self.children.extend(tree.children)
            else:
                self.logger.info('Using existing Trees...')
                progenitor = self.model_progenitor(parent)
                self.logger.info('Progenitor of parent is {}'.format(progenitor))
                if progenitor in self.trees:
                    tree = self.trees.get(progenitor)
                    self.logger.info('Progenitor {} exists in tree history'.format(progenitor))
                elif Tree._id(parent) in self.trees:
                    tree = self.trees.get(Tree._id(parent))
                    self.logger.info('Parent same as progenitor'.format(progenitor))
                else:
                    tree = None
                if tree is not None:
                    self.logger.info('Tree object retrieved ({})'.format(tree))
                    #if self._parent_was_assigned and len(self._seed_params) > 0:
                    if not self._parent_was_assigned: #self._seed_params is not None:
                        change_params = True
                        print("A: Setting change_params =", change_params)
                    else:
                        change_params = False
                        print("B: Setting change_params =", change_params)
                    #print("changed_params =", changed_params)
                    did_set_parent = tree.set_parent(parent, progenitor=False, change_params=change_params)
                    print("Was a parameter set?", did_set_parent)
                    if did_set_parent:
                        print("Did set parent True A")
                        self._parent_was_assigned = True
                        self._seed_params = None
                        self.parents_final.append(parent)
                    #self.logger.info('Parent {} assigned to Tree {}'
                    #                 '\nParent parameters:'
                    #                 '\n{}'.format(Tree._id(parent), tree, Tree.get_params(parent).to_dict()["value"]))
                    self.logger.info('Generating children...')
                    self._generate_children(tree, params=params, n=n_children, i_gen=i_gen, p=p, dfrac=dfrac)
                    self.children.extend(tree.children)
                else:
                    self.logger.info('{} not in tree history {}\n'
                                     'NOTE: Did not generate new children for parent {}'.format(progenitor,
                                                                                                list(self.trees.keys()),
                                                                                                Tree._id(parent)))
        self._i_gen = i_gen
        #return self.stop()

    def stop(self, i_gen=None):
        if self._max_generations():
            return True

    def _max_generations(self, i_gen=None):
        if i_gen is not None:
            self._i_gen = i_gen
        if self._i_gen == self.generations:
            self.logger.info("Maximum number of generations reached ({}/{})".format(self._i_gen, self.generations))
            return True
        else:
            return False

    def get_children(self):
        return self.children


class Norm(object):

    def __init__(self):

        self.logger = logging.getLogger('NOVA.optimize.Norm')
        self.logger.info('creating an instance of Norm')

        self.norm = OrderedDict()

    def add_norm(self, norm=None, lines=None, norm_dict=None):
        """Add normalizing line and the lines it should normalize"""
        if norm_dict is None:
            #print(norm)
            #print(lines)
            if norm is not None:
                self.norm.update({norm: lines})
                #print(self.norm)
        else:
            self.norm = norm_dict

    @staticmethod
    def line_exists(lines, data):
        fail = False
        for l in lines:
            if l not in data.index:
                print("{} not in model.".format(l))
                fail = True
        if fail:
            return False
        else:
            return True

    def _norm_exists(self, data):
        """Check if norm is a valid line"""
        if self.norm:
            return self.line_exists(self.norm, data)
        else:
            print("No normalizing line specified.")
            return False

    def normalize(self, data, norm=None):
        # Check if lines to normalize with exists in data, return exception i
        if norm is None:
            norm = self.norm
        #print(self.norm)
        #if self._norm_exists(data):
        if self.line_exists(norm, data):
            if norm: #self.norm:
                normalized_data = data.copy()
                for n in norm: #self.norm:
                    if norm.get(n) in (None, "all"):  # self.norm.get(n) in (None, "all"):
                        norm_val = data.value.loc[n]
                        if "err" in data.columns:
                            norm_err = data.err.loc[n]
                            normalized_data.loc[:, "err"] = np.abs(data.value / norm_val) * np.sqrt(
                                (data.err / data.value) ** 2 + (norm_err / norm_val) ** 2)
                            normalized_data.loc[:, "value"] = data.value / norm_val  # data.value.loc[n]
                        else:
                            normalized_data.loc[:, "value"] = data.value / norm_val  # data.value.loc[n]
                    else:
                        if self.line_exists(norm.get(n), data):  # self.line_exists(self.norm.get(n), data):
                            norm_val = data.value.loc[n]
                            if "err" in data.columns:
                                norm_err = data.err.loc[n]
                                normalized_data.loc[norm.get(n), "err"] = np.abs(data.value.loc[norm.get(n)]/ \
                                                                                      norm_val) * np.sqrt(
                                    (data.err.loc[norm.get(n)] / data.value.loc[norm.get(n)]) ** 2 +
                                    (norm_err / norm_val) ** 2)
                                normalized_data.loc[norm.get(n), "value"] = data.value.loc[norm.get(n)] / \
                                                                                 norm_val  # data.value.loc[n]
                            else:
                                normalized_data.loc[norm.get(n), "value"] = data.value.loc[norm.get(n)] / \
                                                                                 norm_val  # data.value.loc[n]
                        else:
                            self.logger.info("Norm line {} not in data".format(n))
                            print("Norm line {} not in data".format(n))
                            #self.logger.info("Not all lines in self.norm could be found in model.")
                            #print("Not all lines in self.norm could be found in model.")
                            #raise Exception("Not all lines in self.norm could be found in model.")
                if not set(list(normalized_data.index)).issubset(list(data.index)):
                    self.logger.info("Output does not contain same lines as input.")
                    print("Output does not contain same lines as input.")
                    return None
                else:
                    return normalized_data
        return None


#N = Norm()
#N.add_norm("H__1__9229A", ["H__1__9229A", "H__1_1094M", "H__1_1282M"])
#N.add_norm("H__1_2166M", ["H__1_2166M", "6LEV_1130M", "NE_2_1281M"])
#N.add_norm("Bla", ["da"])
#N.normalize(obs)
