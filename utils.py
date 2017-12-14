"""
Copyright: Emilia Calven

Contains general useful functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import logging

module_logger = logging.getLogger('NOVA.utils')

def make_dir(_dir, verbose=1):
    """Create directory if it doesn't exist already"""
    import os
    try:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
            if verbose > 0:
                print("\nCreated directory {}".format(_dir))
                module_logger.info("Created directory {}".format(_dir))
        else:
            if verbose > 1:
                print("\n{} already exists.".format(_dir))
    except:
        print("Could not create path {}".format(_dir))


def uuid():
    """Return a Universal Unique ID (UUID)"""
    import uuid
    return uuid.uuid4().hex

def set_makefile_paths(path):
    """Set CLOUDY_EXE AND CLOUDY_DATA_PATH to the Cloudy path
       set in the Makefile.
    """
    try:
        with open(path+"Makefile") as f:
            cloudy_exe_dir = f.readline()
        cloudy_exe_dir = cloudy_exe_dir.split("=")[-1].strip()
        cloudy_dir = cloudy_exe_dir.rsplit("/", 2)[0].split()[-1]
        cloudy_data_path = cloudy_dir + "/data"
        os.putenv("CLOUDY_EXE", cloudy_exe_dir)
        os.putenv("CLOUDY_DATA_PATH", cloudy_data_path)
        #!echo $CLOUDY_EXE
        #!echo $CLOUDY_DATA_PATH
    except:
        raise


def grid_maker(*args):
    # find the sizes of each dimension and the total size of the
    # final array
    shape = [len(arg) for arg in args]
    size = 1
    for sh in shape:
        size *= sh
    # make a list of lists to hold the indices
    l = [1 for i in range(len(args))]
    idx = [l[:] for i in range(size)]
    # fill in the indices
    rep = 1
    for aidx, arg in enumerate(args):
        # repeat each value in the dimension based on which
        # dimensions we've already included
        vals = []
        for val in arg:
            vals.extend([val] * rep)
        # repeat each dimension based on which dimensions we
        # haven't already included and actually fill in the
        # indices
        rest = size / (rep * len(arg))
        #print(vals, rest)
        for vidx, val in enumerate(vals * int(rest)):
            idx[vidx][aidx] = val
        rep *= len(arg)
    return idx


class CalcVol(object):
    def __init__(self, m, a, b=1.):
        self.m = m
        self.a = a
        self.b = b
        # self.vol_mean(self.a, self.b)

    def _quiet_div(self, a, b):
        if a is None or b is None:
            to_return = None
        else:
            np.seterr(all="ignore")
            to_return = a / b
            np.seterr(all=None)
        return to_return

    def integ(self, a):
        """
        Integral of a on the volume (taken from pyCloudy source code)

        Equation:
            vol_integ(a) = \f$\int a.ff.dV\f$

        :param m: Cloudy Model object
        :param a:
        """
        if a is None or self.m.dv is None:
            return None
        else:
            return (a * self.m.dvff).sum()

    @classmethod
    def mean(cls, m, a, b=1.):
        """
        Return the mean value of a weighted by b on the volume
        (taken from pyCloudy source code)

        Equation:
            vol_mean(a, b) = \f$\frac{\int a.b.ff.dV}{\int b.ff.dV}\f$

        :param m:
        """
        cv = cls(m, a, b)
        return cv._quiet_div(cv.integ(a * b), cv.integ(b))
