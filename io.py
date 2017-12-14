import sys
import pyCloudy as pc
from sqlalchemy import create_engine
from nova.datastruct import DataStruct
import pandas as pd
from glob import glob
import random
import logging

# create logger
module_logger = logging.getLogger('NOVA.io')

def load_models(model_name="", path="", read_grains=False, **kwargs):
    """Load in models in path with model_name. Skip broken models or models with 1 zone."""
    models = pc.load_models('{0}{1}'.format(path, model_name), read_grains=read_grains, **kwargs)
    models_return = []
    for m in models:
        if m.n_zones > 1:
            models_return.append(m)
        else:
            # log failed model
            pass
    return models_return


class IO(object):

    def __init__(self, db=None, engine_type="sqlite"):

        self.logger = logging.getLogger('NOVA.io.IO')
        self.logger.info('creating an instance of IO')

        self.db = db
        self.engine_type = engine_type
        self.engine = None

    def connect_to_db(self, db=None, engine=None):
        if self.engine is not None:
            self.engine = engine
        if self.db is None:
            self.db = db
        if self.engine is None:
            try:
                self.engine = create_engine('{}:///{}'.format(self.engine_type, self.db))
            except:
                raise

class ImportExport(object):

    def __init__(self, engine=None):  # abunds, params, path, model_name, emis_file, db):

        self.logger = logging.getLogger('NOVA.io.ImportExport')
        self.logger.info('creating an instance of ImportExport')

        self.engine = engine
        if self.engine is None:
            self.engine_url = None
        else:
            self.engine_url = self.engine.url
        self._models = None
        self.tables_list = []
        self.id = None
        self.previous_models = []
        self.failed_models = []
        self.exported_models = []
        # self._mod_list = None

    @staticmethod
    def _model_failed(model):
        if model.n_zones > 1:
            return False
        else:
            return True

    @staticmethod
    def add_params_attr(models):
        models_return = []
        for model in models:
            try:
                model_path = model.model_name.rsplit("/", 1)[0] + "/"
                model_id = model.model_name_s.split("_")[0]
                file = glob(model_path + "{}*.pars".format(model_id))
                if file:
                    df = pd.read_csv(file[0], sep="\t")
                    params = {par: val for par, val in zip(df["param"], df["value"])}
                    model.__setattr__("params", params)
                else:
                    model.__setattr__("params", None)
                models_return.append(model)
            except:
                raise
        return models_return

    @staticmethod
    def add_progenitor(models):
        models_return = []
        for model in models:
            try:
                model_path = model.model_name.rsplit("/", 1)[0] + "/"
                model_id = model.model_name_s.split("_")[0]
                file = glob(model_path + "{}*.prog".format(model_id))
                if file:
                    with open(file[0]) as f:
                        progenitor = f.readline()
                    model.__setattr__("progenitor", progenitor)
                else:
                    model.__setattr__("progenitor", None)
                models_return.append(model)
            except:
                raise
        return models_return

    @staticmethod
    def add_extras(models):
        models_return = []
        for model in models:
            try:
                model_path = model.model_name.rsplit("/", 1)[0] + "/"
                model_id = model.model_name_s.split("_")[0]
                file = glob(model_path + "{}*.ext".format(model_id))
                if file:
                    df = pd.read_csv(file[0], sep=",")
                    for col in df.columns:
                        model.__setattr__(col, df[col].values[0])
                else:
                    pass
                models_return.append(model)
            except:
                raise
        return models_return

    @staticmethod
    def _format_table(row):
        """Recast to correct types"""
        row = pd.to_numeric(row, errors="ignore")
        #try:
        #    row = eval(row)
        #except (TypeError, ValueError, SyntaxError, NameError):
        #    #print("Could not evaluate {}".format(row))
        #    pass
        if not row == row:
            row = None
        return row

    def format_table(self, list_):
        return_list = []
        for l in list_:
            df = l["model_params"]
            df["value"] = df["value"].apply(self._format_table)
            l["model_params"] = df
            return_list.append(l)
        return return_list  # [l["value"].apply(self._format_table) for l in list_]

    @staticmethod
    def _pc_load_models(model_name = None, mod_list = None, n_sample = None, verbose = False, **kwargs):
        """
            Return a list of CloudyModel correspondig to a generic name

            Parameters:
                - model_name:    generic name. The method is looking for any "model_name*.out" file.
                - mod_list:      in case model_name=None, this is the list of model names (something.out or something)
                - n_sample:      randomly select n_sample from the model list
                - verbose:       print out the name of the models read
                - **kwargs:      arguments passed to CloudyModel
            """

        if model_name is not None:
            mod_list = glob.glob(model_name + '*.out')
        if mod_list is None or mod_list == []:
            pc.log_.error('No model found', calling='load models')
            return None
        if n_sample is not None:
            if n_sample > len(mod_list):
                pc.log_.error('less models {0:d} than n_sample {1:d}'.format(len(mod_list), n_sample),
                              calling='load models')
                return None
            mod_list = random.sample(mod_list, n_sample)
        m = []
        for outfile in mod_list:
            if outfile[-4::] == '.out':
                model_name = outfile[0:-4]
            else:
                model_name = outfile
            try:
                cm = pc.CloudyModel(model_name, verbose=0, **kwargs)
                if not cm.aborted:
                    m.append(cm)
                if verbose:
                    print('{0} model read'.format(outfile[0:-4]))
            except:
                pass
        pc.log_.message('{0} models read'.format(len(m)), calling='load_models')
        return m

    def load_models(self, path=None, model_name="", read_grains=False, mod_list=None, n_models=None, **kwargs):
        """Load in models in path with model_name. Skip broken models or models with 1 zone."""

        if mod_list is None:
            mod_list = glob('{0}{1}'.format(path, model_name) + '*.out')
            if mod_list is not None:
                mod_list = mod_list[0:n_models]
            mod_list = list(set(mod_list).difference(self.previous_models))
            mod_list = list(set(mod_list).difference(self.failed_models))
        # self._mod_list = mod_list
        #self.logger.info("Loading models {}...".format(mod_list))
        #self._models = pc.load_models('{0}{1}'.format(path, model_name), mod_list=mod_list, read_grains=read_grains,
        #                              **kwargs)
        #sys.stdout.write(len(mod_list))
        #self.logger.info("model list:")
        #self.logger.info(mod_list)
        if len(mod_list) > 0:
            self.logger.info("Loading models {}...".format(mod_list))
            # self._models = pc.load_models(None, mod_list=mod_list, read_grains=read_grains, **kwargs)
            self._models = self._pc_load_models(None, mod_list=mod_list, read_grains=read_grains, **kwargs)
            self._models = self.add_params_attr(self._models)
            self.previous_models.extend([m.model_name + ".out" for m in self._models])
            return True
        else:
            self.logger.info("No models to load...")
            print("No models to load...")
            return False
        #self._models = self.add_progenitor(self._models)
        #self._models = self.add_extras(self._models)
        #self.previous_models.extend(mod_list)

    def to_sql(self, export=True, **kwargs):
        # Reset tables_list list
        self.tables_list = []

        if self.engine is not None:
            if self._models is not None:
                for m in self._models:
                    if not self._model_failed(m):
                        # Build model data structure
                        data = DataStruct(m)
                        data.build()
                        tables = data.df_tables
                        self.tables_list.append(tables)
                        if export:
                            export_success = self._export(tables, **kwargs)
                            #if export_success:
                            self.exported_models.append(m.model_name)
                    else:
                        self.failed_models.append(m.model_name + ".out")
                        print("Failed model.")
            else:
                print("No models to export.")
        else:
            print("No database specified. Did not export models.")

    def _export(self, tables, if_exists="append", index=False, **kwargs):
        if not self._exists(tables):
            for name, table in tables.items():
                table.to_sql(name, self.engine, index=index, if_exists=if_exists, **kwargs)
            self._print_success()
            return True
        else:
            #print("Model already exists. Skipping...")
            return False

    def _print_success(self):
        print("Model {} successfully exported to {}.".format(self.id, self.engine_url))
        self.logger.info("Model {} successfully exported to {}.".format(self.id, self.engine_url))

    def _exists(self, tables):
        self.id = tables["model_info"]["id"].values[0]
        if "model_info" in self.engine.table_names():
            res = self.engine.execute("select id from model_info where id = '{}'".format(self.id))
            if res.first() is not None:
                return True
        else:
            return False

    @staticmethod
    def read_sql(self):
        """Method to read tables from SQL database."""
        pass
