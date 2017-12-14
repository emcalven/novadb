import numpy as np
import re
import pandas as pd
from nova.utils import CalcVol

import logging

# create logger
module_logger = logging.getLogger('NOVA.datastruct')

class DataStruct(object):

    def __init__(self, model):

        """
        :param model: pyCloudy Model object
        """

        self.logger = logging.getLogger('NOVA.datastruct.DataStruct')
        self.logger.info('creating an instance of DataStruct')

        self.model = model
        self._model_id = self.model.model_name_s.split("_")[0]
        self._model_info = {}
        self._lines = {"line_id": [], "value": [], "model_id": []}
        self._ionic_frac = {"model_id": [], "ion": [], "value": []}
        self._abundance = {"model_id": [], "elem": [], "value": []}
        self._model_input = {"model_id": None, "input": None}
        self._params = {"model_id": [], "param": [], "value": []}
        self.df_tables = {"model_info": None, "lines": None, "ionic_frac": None, "abund": None, "model_input": None}

    def build(self):
        self._tab_model_info()
        self._tab_lines()
        self._tab_ionic_frac()
        self._tab_abundance()
        self._tab_model_input()
        self._tab_model_params()
        self._make_df()

    def _make_df(self):
        self.df_tables = {"model_info": pd.DataFrame(self._model_info, index=[0]),
                          "lines": pd.DataFrame(self._lines),
                          "ionic_frac": pd.DataFrame(self._ionic_frac),
                          "abund": pd.DataFrame(self._abundance),
                          "model_input": pd.DataFrame(self._model_input, index=[0]),
                          "model_params": pd.DataFrame(self._params)}

    def _tab_model_info(self):
        # model_info table
        self._model_info["user"] = self.model.params.get("progenitor")  # self.model.user
        self._model_info["project"] = self.model.params.get("project")  # self.model.project
        self._model_info["dir"] = self.model.model_name.rsplit("/", 1)[0]
        self._model_info["file"] = self.model.model_name_s
        self._model_info["id"] = self._model_id
        self._model_info["cloudy_version"] = self.model.cloudy_version
        self._model_info["rad_field"] = self.model.out.get("Blackbody").split()[1]
        self._model_info["rad_temp"] = self.model.Teff

        if self.model.out["SED1"].strip().startswith("L"):
            self._model_info["lumi_unit"] = "erg s-1"
        elif self.model.out["SED1"].strip().startswith("I"):
            self._model_info["lumi_unit"] = "erg s-1 cm-2"

        for i in range(1, 8):
            try:
                self._model_info["rad_field{}".format(i)] = self.model.out["SED{}".format(i)].strip()
            except:
                self._model_info["com{}".format(i)] = None

        for i in range(1, 5):
            try:
                self._model_info["stop_crit{}".format(i)] = self.model.params.get("stop_crit")[i-1]
            except:
                self._model_info["stop_crit{}".format(i)] = None

        self._model_info["lumi"] = np.float(self.model.out["SED1"].strip().split()[1])
        self._model_info["dens"] = self.model.nH[0]
        self._model_info["dens_power"] = self.model.params.get("nh_power")
        self._model_info["r_in"] = self.model.r_in
        self._model_info["r_out"] = self.model.r_out
        self._model_info["ff"] = self.model.ff[0]
        self._model_info["cf"] = self.model.params.get("cf")

        for i in range(1, 10):
            try:
                self._model_info["com{}".format(i)] = self.model.comment[i-1]
            except:
                self._model_info["com{}".format(i)] = None

        self._model_info["distance"] = self.model.distance
        self._model_info["thickness"] = self.model.thickness
        self._model_info["n_zones"] = self.model.n_zones
        self._model_info["CloudyEnds"] = self.model.out.get("Cloudy ends").strip()
        self._model_info["CalculStop"] = self.model.out.get("stop").strip()
        self._model_info["te_in"] = self.model.te[0]
        self._model_info["te_out"] = self.model.te[-1]
        self._model_info["te_mean"] = CalcVol.mean(self.model, self.model.te)  # m.te.mean()
        self._model_info["ne_in"] = self.model.ne[0]
        self._model_info["ne_out"] = self.model.ne[-1]
        self._model_info["ne_mean"] = CalcVol.mean(self.model, self.model.ne)  # m.ne.mean()
        self._model_info["H_mass"] = self.model.H0_mass
        self._model_info["H1_mass"] = self.model.H_mass
        self._model_info["nH_in"] = self.model.nH[0]
        self._model_info["nH_out"] = self.model.nH[-1]
        self._model_info["nH_mean"] = self.model.nH_mean
        self._model_info["theta"] = self.model.theta
        self._model_info["progenitor"] = self.model.params.get("progenitor")  # self.model.progenitor
        self._model_info["datetime"] = self.model.date_model
        self._model_info["exec_time"] = float(self.model.out.get("Cloudy ends").rsplit("ExecTime(s)")[-1].split("\n")[0].
                                              strip())
        self._model_info["v_min"] = self.model.params.get("v_min")
        self._model_info["v_max"] = self.model.params.get("v_max")
        self._model_info["exp_time"] = self.model.params.get("exp_time")

    def _tab_lines(self):
        # lines table
        for l in self.model.emis_labels:
            self._lines["line_id"].append(l)
            self._lines["value"].append(self.model.get_emis_vol(l))
            self._lines["model_id"].append(self._model_id)

    def _tab_ionic_frac(self):
        # ionic_frac table
        for elem in self.model.ionic_names:
            for ion in self.model.ionic_names[elem]:
                elem_str = re.findall('\d+|\D+', ion)[0]
                if len(elem_str) == 1:
                    continue
                if len(elem_str.replace("__", "").replace("_", "")) > 1 and elem_str.isupper():  # and word.isalpha()
                    continue
                ion_str = ion.replace('__', '').replace('_', '')
                ion_get = re.findall('\d+|\D+', ion_str)

                self._ionic_frac["model_id"].append(self._model_id)
                self._ionic_frac["ion"].append(ion_get[0] + "_" + str(int(ion_get[1]) - 1))
                self._ionic_frac["value"].append(self.model.get_ab_ion_vol(ion_get[0], int(ion_get[1]) - 1))

    def _tab_abundance(self):
        # abundance table
        for ab in self.model.abund:
            self._abundance["model_id"].append(self._model_id)
            self._abundance["elem"].append(ab)
            self._abundance["value"].append(self.model.abund[ab])

    def _tab_model_input(self):
        # model_input table
        with open(self.model.model_name + ".in") as f:
            self._model_input["model_id"] = self._model_id
            inp = ""
            for l in f.readlines():
                inp += l
            self._model_input["input"] = inp

    def _tab_model_params(self):
        # model_input table
        for par, val in self.model.params.items():
            self._params["model_id"].append(self._model_id)
            self._params["param"].append(par)
            self._params["value"].append(val)
