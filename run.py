import pyCloudy as pc
import sys
sys.path.append("../../")
from nova.model import Cloudy13, Cloudy17
from nova.gridmaker import get_grid #grid_maker, set_makefile_paths
from nova.datastruct import DataStruct
from nova.io import load_models #, ExportToSQL
import numpy as np

import logging

# create logger
module_logger = logging.getLogger('NOVA.run')

class MakeRun(object):

    def __init__(self, path, version, iterator=None):

        self.logger = logging.getLogger('NOVA.run.MakeRun')
        self.logger.info('creating an instance of MakeRun')

        self.path = path
        self.version = version
        self.db = None
        self.iterator = iterator

    def make_input(self, model_name, grid, emis_file=None, uuid=None, version="c13"):
        for gp in grid:
            if self.iterator is not None:
                uuid = self.iterator

            #if self.version == "c13":
            #    cloudy = Cloudy13(abund=gp["abund"], emis_file=emis_file, path=self.path, model_name=model_name,
            #                      params=gp, progenitor=gp["progenitor"], user=gp["user"], project=gp["project"],
            #                      uuid=uuid, extras=gp["extras"], **gp["params"])
            #elif self.version == "c17":
            #    cloudy = Cloudy17(abund=gp["abund"], emis_file=emis_file, path=self.path, model_name=model_name,
            #                      params=gp, progenitor=gp["progenitor"], user=gp["user"], project=gp["project"],
            #                      uuid=uuid, extras=gp["extras"], **gp["params"])

            #if self.version == "c13":
            #    cloudy = Cloudy13(abund=gp["abund"], emis_file=emis_file, path=self.path, model_name=model_name,
            #                      params=gp,
            #                      uuid=uuid, extras=gp["extras"], **gp["params"])
            #elif self.version == "c17":
            #    cloudy = Cloudy17(abund=gp["abund"], emis_file=emis_file, path=self.path, model_name=model_name,
            #                      params=gp,
            #                      uuid=uuid, extras=gp["extras"], **gp["params"])

            if self.version == "c13":
                cloudy = Cloudy13(abund=gp["abund"], emis_file=emis_file, path=self.path, model_name=model_name,
                                  params=gp,
                                  _uuid=uuid, **gp["params"])
            elif self.version == "c17":
                cloudy = Cloudy17(abund=gp["abund"], emis_file=emis_file, path=self.path, model_name=model_name,
                                  params=gp,
                                  _uuid=uuid, **gp["params"])

            #print("1----------------------1")
            #print(gp)
            if self.iterator is not None:
                self.iterator += 1

            cloudy.set_all()
            #cloudy.print_input(verbose=False)
            cloudy.print_all(verbose=False)
        #!echo $CLOUDY_EXE
        #!echo $CLOUDY_DATA_PATH

    def run_cloudy(self, model_name="", use_make=True, n_proc=2):
        pc.run_cloudy(dir_=self.path, n_proc=n_proc, model_name=model_name, use_make=use_make)





if __name__ == '__main__':

    cloudy_version = "c13"
    iterate = True

    # Connect to Database
    db = ExportToSQL(db=":memory:")
    db.connect()

    # Set parameters
    params = {}
    abunds = {}
    params["teff"] = np.array([10 ** 5])  # , 10**5, 10**6])
    params["lum"] = np.array([34.74])
    params["nh"] = np.array([1e6])  # , 1e6])
    params["r_in"] = np.array([10 ** 15.59])
    params["r_out"] = np.array([10 ** 15.63])

    abunds["N"] = np.array([1])
    abunds["O"] = np.array([35])
    abunds["Ne"] = np.array([0.5])
    abunds["Mg"] = np.array([180])

    # Define path and model name
    path = "./full_chain/c13/"
    model_name = "full_chain_c13"
    emis_file = "../data/poster_lines_c13.dat"


    #while iterations:

    # DON'T TOUCH CODE BELOW THIS LINE
    # ------------------------------------------------------------------------------------

    # Parameter grid
    grid = get_grid(abunds, params)

    # Make input files and run Cloudy
    mr = MakeRun(path=path, version=cloudy_version)
    mr.make_input(model_name=model_name, grid=grid, emis_file=emis_file)
    mr.run_cloudy()

    # Load models and export to database
    #ie = ImportExport(path, db)
    #ie.load()
    #ie.export_to_sql()

    # Compare predict to obs
    #if iterate:

    #compare_result = []
    #if iterate:
    #    pred = normed_pred(tables["lines"])
    #    if obs is not None:
    #        res = chisq(obs, pred)
    #    else:
    #        res = None
    #    compare_result.append(res)