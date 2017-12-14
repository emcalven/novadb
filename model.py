import os
import pyCloudy as pc
from pyCloudy.utils.init import SYM2ELEM
from nova.utils import make_dir, uuid
import numpy as np
from numbers import Number
import warnings
import pandas as pd

import logging

# create logger
module_logger = logging.getLogger('NOVA.model')

class Model(object):
    def __init__(self, teff=1e5, lumi_unit="luminosity total", lumi_val=35, nh=None, nh_power=None, ff=1, cf=1, r_in=None,
                 r_out=None, stop_crit=None, verbose=1, abund_ref="GASS10", abund={}, abund_type="scale", grains=None,
                 cr=True, cmb=True, sphere=True, molecules=False, theta=None, emis_file=None,
                 path=None, model_name="foo", other_opt=[], unique=True, _uuid=None, params={}, progenitor=None,
                 user=None, project=None, extras={}, **kwargs):

        """
        Module to help define and create consistent Cloudy input files.

        :param teff: int, float: Temperature of source blackbody [K]
        :param lumi_unit: str: Cloudy luminosity unit (e.g. "luminosity total")
        :param lumi_val: int, float: Luminosity value [log cm-3]
        :param nh: int, float: Hydrogen density (cm-3)
        :param nh_power: int, float: Index of hydrogen power law (optional)
        :param ff: int, float: Filling factor
        :param cf: int, float: Cover factor
        :param r_in: int, float: Inner radius of nebula [cm]
        :param r_out: int, float: Outer radius of nebula [cm] (optional)
        :param stop_crit: str, list(str): Stop criteria (optional)
        :param verbose: int: pyCloudy verbosity level (optional)
        :param abund_ref: str: Cloudy built-in abundances
        :param abund: dict: Dictionary with element : abundance (e.g. {He:2, O:3})
        :param abund_type: str:
        :param grains:
        :param cr:
        :param cmb:
        :param sphere:
        :param molecules: (False, "H2")
        :param theta:
        :param emis_file:
        :param path:
        :param model_name:
        :param other_opt:
        :param kwargs:
        """

        self.logger = logging.getLogger('NOVA.model.Model')
        self.logger.info('creating an instance of Model')

        if "cloudy_exe" in kwargs:
            os.putenv("CLOUDY_EXE", kwargs["cloudy_exe"])
            pc.config.cloudy_exe = kwargs["cloudy_exe"]
        if "cloudy_data_path" in kwargs:
            os.putenv("CLOUDY_DATA_PATH", kwargs["cloudy_data_path"])

        self.pc = pc
        self.teff = teff
        self.lumi_unit = lumi_unit
        self.lumi_val = lumi_val
        self.nh = nh
        self.nh_power = nh_power
        self.ff = ff
        self.cf = cf
        self.r_in = r_in
        self.r_out = r_out
        self.stop_crit = stop_crit
        self.abund_ref = abund_ref
        self.abund = abund
        self.abund_type = abund_type
        self.grains = grains
        self.cr = cr
        self.cmb = cmb
        self.sphere = sphere
        self.molecules = molecules
        self.theta = theta
        self.emis_file = emis_file
        self.path = path
        self.verbose = verbose
        self.kwargs = kwargs
        self.use_cste_density = True
        self.options = []
        self.model_name = model_name
        self.other_opt = other_opt
        self.params = params
        self.progenitor = progenitor  # Belongs to extras (print_extras())
        self.user = user #extras.get("user")  # Belongs to extras (print_extras())
        self.project = project #extras.get("project")  # Belongs to extras (print_extras())
        #self.user = user  # Belongs to extras (print_extras())
        #self.project = project  # Belongs to extras (print_extras())
        self.extras = extras
        self._uuid = _uuid

        if not self.path.endswith("/"):
            self.path += "/"
        make_dir(self.path, verbose=1)  # Creates directory to place models in
        if unique:
            if self._uuid is None:
                self._uuid = uuid()
                self.model_name_full = self.path + self._uuid + "_" + self.model_name
                self.c_input = pc.CloudyInput(self.model_name_full)
            else:
                if isinstance(self._uuid, (int, float)):
                    self._uuid = str(self._uuid)
                self.model_name_full = self.path + self._uuid + "_" + self.model_name
                self.c_input = pc.CloudyInput(self.model_name_full)
        else:
            self.model_name_full = self.path + self.model_name
            #self.model_name_full = self.path + self._uuid + "_" + self.model_name
            self.c_input = pc.CloudyInput(self.model_name_full)

    # def print_genes(self):
    #
    #
    #     {'uuid': '2',
    #      'abund': {},
    #      'abund_ref': 'GASS10',
    #      'abund_type': 'scale',
    #     'cf': 1,
    #     'cmb': True,
    #     'cr': True,
    #     'emis_file': None,
    #     'extras': {},
    #     'ff': 1,
    #     'grains': None,
    #     'lumi_unit': 'luminosity total',
    #     'lumi_val': 35,
    #     'model_name': 'foo',
    #     'model_name_full': 'Bla/2_foo',
    #     'molecules': False,
    #     'nh': None,
    #     'nh_power': None,
    #     'options': [],
    #     'other_opt': [],
    #     'params': {},
    #     'path': 'Bla/',
    #     'progenitor': None,
    #     'project': None,
    #     'r_in': None,
    #     'r_out': None,
    #     'sphere': True,
    #     'stop_crit': None,
    #     'teff': 100000.0,
    #     'theta': None,
    #     'use_cste_density': True,
    #     'user': None,
    #     'verbose': 1}

    def set_all(self):
        self._luminosity()
        self._hydrogen_density()
        self._theta()
        self._radius()
        self._abundance_reference()
        self._grains()
        self._options()
        self._sphere()
        self._iterate()
        self._stop_criteria()
        self._emission_lines()

    def _luminosity(self):
        # Define ionizing SED: Effective temperature and luminosity.
        # lumi_unit is one of the Cloudy options, e.g. "luminosity total", "q(H)"
        self.c_input.set_BB(Teff=self.teff, lumi_unit=self.lumi_unit, lumi_value=self.lumi_val)

    def _options(self):
        self._abundance()
        self._cosmic_rays()
        self._cmb()
        self._molecules()
        self._covering_factor()
        self._other_options()
        # Set them all on pyCloudy model object
        self.c_input.set_other(self.options)

    # Option
    def _abundance(self):
        # Elemental abundances
        if self.abund_type == "scale":
            abund_cmd = "element scale factor"
        # TODO: Add more ways to set abundances in elif statements...
        # Default to:
        elif isinstance(self.abund_type, str):
            abund_cmd = "element " + self.abund_type
            print("Abundances are given with the command {}.".format(self.abund_type))
        else:
            abund_cmd = "element scale factor"
            print("Abundances are given as scale factors of {}.".format(self.abund_ref))
        if self.abund:
            for elem, elem_val in self.abund.items():
                if SYM2ELEM.get(elem) is not None:
                    elem = SYM2ELEM.get(elem)
                else:
                    print("CAUTION: Is {} correct element assignment?".format(elem))
                self.options.append("{} {} {}".format(abund_cmd, elem, elem_val))

    # Option
    def _cosmic_rays(self):
        # Cosmic rays
        if self.cr:
            self.options.append('COSMIC RAY BACKGROUND')

    # Option
    def _cmb(self):
        # Cosmic microwave background
        if self.cmb:
            if isinstance(self.cmb, int):
                self.options.append('CMB, {}'.format(self.cmb))
            else:
                self.options.append('CMB')

    # Option
    def _molecules(self):
        # Molecules on/off
        # self.molecules = (False, "H2") turns off H2 molecules
        if isinstance(self.molecules, bool):
            if not self.molecules:
                self.options.append('no molecules'.format())
        elif isinstance(self.molecules, (tuple, list)):
            if not self.molecules[0]:
                try:
                    self.options.append('no {} molecules'.format(self.molecules[1]))
                except ValueError as e:
                    print("Input error({0}): {1}".format(e.errno, e.strerror))

    # Option
    def _covering_factor(self):
        # Covering factor
        self.options.append('covering factor {}'.format(self.cf))

    # Option
    def _other_options(self):
        # Other options
        if isinstance(self.other_opt, np.ndarray):
            self.other_opt = str(list(self.other_opt))
        #print("1----------------------1")
        #print(self.other_opt)
        #print(type(self.other_opt))
        if self.other_opt:
            #print("2----------------------2")
            #print(self.other_opt)
            if not isinstance(self.other_opt, list):
                self.other_opt = [self.other_opt]
            # print("1----------------------1")
            # print(self.other_opt)
            # print(type(self.other_opt))
            for other in self.other_opt:
                #print(other)
                try:
                    other = eval(other)
                    #print(other)
                    #print(type(other))
                except:
                    pass
                self.options.append(other)
                # Special hydrogen density command turns off default
                if 'hden' in other:
                    self.use_cste_density = False

    def _hydrogen_density(self):
        # Define density. You may also use set_dlaw(parameters) if you have a density
        # law defined in dense_fabden.cpp.
        if self.use_cste_density:
            self.c_input.set_cste_density(np.log10(self.nh), ff=self.ff)
            if self.nh_power is not None:
                if not isinstance(self.nh_power, (int, float)):
                    raise ValueError("power has to be a number.")
                else:
                    # Not pretty but only way to set a power law when using set_cste_density() w/o modding the source
                    self.c_input._density += ', power = {0:.3f}'.format(self.nh_power)

    def _theta(self):
        # For 3D models, set angle theta
        if self.theta is not None:
            if not isinstance(self.theta, (int, float)):
                raise ValueError("theta has to be a number.")
            elif self.theta < 0. or self.theta > 180.:
                raise ValueError("theta has to satisfy 0 <= theta <= 180.")
            else:
                self.c_input.set_theta_phi(self.theta)

    def _radius(self):
        # Define inner (and outer) radius. If outer radius defined we have a matter-bounded nebula.
        if self.r_out is not None:
            if not isinstance(self.r_out, (int, float)):
                raise ValueError("r_out has to be a number.")
            else:
                r_out = np.log10(self.r_out)
        else:
            r_out = None
        if self.r_in is not None:
            if not isinstance(self.r_in, (int, float)):
                raise ValueError("r_in has to be a number.")
            else:
                self.c_input.set_radius(r_in=np.log10(self.r_in), r_out=r_out)
        #if self.r_in is not None:
        #    self.c_input.set_radius(r_in=r_in, r_out=r_out)

    def _abundance_reference(self):
            self.c_input.set_abund(ab_dict=None, predef=self.abund_ref, nograins=True)

    def _grains(self):
        if self.grains is not None:
            try:
                self.grains = '"{}"'.format(self.grains)
            except:
                pass
            self.c_input.set_grains(grains=self.grains)

    def _iterate(self):
        self.c_input.set_iterate(
            to_convergence=True)  # (0) for no iteration, () for one iteration, (N) for N iterations.

    def _sphere(self):
        self.c_input.set_sphere(self.sphere)  # () or (True) : sphere, or (False): open geometry.

    # TODO: Include file with emission lines (should be same for all models, but can be changed if needed)
    def _emission_lines(self):
        if self.emis_file is None:
            warnings.warn("Oops, I have no emission lines!")
        else:
            self.c_input.read_emis_file(self.emis_file)

    def _stop_criteria(self):
        self.c_input.set_stop(stop_criter=self.stop_crit)  # Sets the stop criteria

    def print_input(self, verbose=True):
        # Writing the Cloudy inputs. to_file for writing to a file (named by full_model_name). verbose to print
        # on the screen.
        self.c_input.print_input(to_file=True, verbose=verbose)  # , opt='density')
        self.c_input.print_make_file()

    def print_params(self):
        param_dict = {"param": [], "value": []}
        for key in self.params:
            if key not in ("bla"): #("progenitor", "user", "project", "extras"):
                for par, val in self.params[key].items():
                    if True: #isinstance(val, Number):
                        param_dict["param"].append(par)
                        param_dict["value"].append(val)
        pd.DataFrame(param_dict).to_csv(self.model_name_full + ".pars", index=False, sep="\t")

    def print_progenitor(self):
        with open(self.model_name_full + ".prog", "w") as f:
            if self.progenitor is not None:
                f.write(self.progenitor)
            else:
                f.write("Unknown")

    def print_extras(self):
        #extras_dict = {"progenitor": self.progenitor, "user": self.user, "project": self.project}
        pd.DataFrame(self.extras, index=[0]).to_csv(self.model_name_full + ".ext", index=False, sep=",")
        #pd.DataFrame({})
        #with open(self.model_name_full + ".ext", "w") as f:
        #    f.write(self.progenitor)
        #    f.write(self.user)
        #    f.write(self.project)

    def print_all(self, verbose=True):
        self.print_input(verbose=verbose)
        self.print_params()
        self.print_progenitor()
        #self.print_extras()


class Cloudy13(Model):

    def __init__(self, emis_file=None, **kwargs):
        cloudy_exe = "/usr/local/Cloudy/c13.05/source/cloudy.exe"
        cloudy_data_path = "/usr/local/Cloudy/c13.05/data"
        if emis_file is None:
            emis_file = "../data/c13_lines.dat"

        super().__init__(**kwargs, cloudy_exe=cloudy_exe, cloudy_data_path=cloudy_data_path,
                         emis_file=emis_file)


class Cloudy17(Model):

    def __init__(self, emis_file=None, **kwargs):
        cloudy_exe = "/usr/local/Cloudy/c17.00/source/cloudy.exe"
        cloudy_data_path = "/usr/local/Cloudy/c17.00/data"
        if emis_file is None:
            emis_file = "../data/c17_lines.dat"

        super().__init__(**kwargs, cloudy_exe=cloudy_exe, cloudy_data_path=cloudy_data_path,
                         emis_file=emis_file)
