from pyCloudy.utils.init import SYM2ELEM
import logging

module_logger = logging.getLogger('NOVA.gridmaker')


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

def reshape_grid(grid):
    store = []
    for row in grid:
        row_dict = {}
        row_list = []
        for col in row:
            if type(col) is dict:
                for element in col.keys():
                    row_dict[element] = col[element]
            else:
                row_list.append(col)
        if row_dict:
            row_list.append(row_dict)
        store.append(row_list)
    return store

"""def get_grid(abund_grid, element_list, *args):
    all_args = [abund_grid[element] for element in element_list]
    if args:
        for arg in args:
            all_args.append(arg)
        grid = grid_maker(*tuple(all_args))
        #grid = mm.grid_maker(args, *tuple(elem))
    else:
        grid = grid_maker(*tuple(all_args))
    return reshape_grid(grid)"""

def structured_grid(grid):
    comb = []
    for gp in grid:
        A = {}
        B = {}
        C = None
        D = None
        #E = None
        for par in gp:
            key = list(par.keys())[0]
            if key in list(SYM2ELEM.keys()):
                A[key] = par[key]
            #elif key == "progenitor":
            #    C = par[key]
            #elif key == "extras":
            #    D = par[key]
            #elif key == "user":
            #    D = par[key]
            #elif key == "project":
            #    E = par[key]
            else:
                B[key] = par[key]
        #D = {"abund": A, "params": B, "progenitor": C, "user": D, "project": E}
        D = {"abund": A, "params": B}#, "progenitor": C#, "extras": D}
        comb.append(D)
    return comb


def get_grid(params=None, is_generated=False):
    if not is_generated:
        if params is not None:
            par_grid = prep_grid(params)
        else:
            return False
        if par_grid is not None:
            # params = [par_grid.get(par) for par in par_grid if par_grid.get(par) is not None]
            params = [par_grid.get(par) for par in par_grid]

        grid = grid_maker(*tuple(params))
    elif is_generated:
        #print(params)
        # grid = [[{par: par_grid.get(par)} for par in par_grid if par_grid.get(par) is not None] for par_grid in params]
        grid = [[{par: par_grid.get(par)} for par in par_grid] for par_grid in params]
        #print(grid)
    return structured_grid(grid)

def prep_grid(D):
    """
    :param D: Dict of dicts
    :return:
    """
    try:
        return {key:[{key:val} for val in D[key]] for key in D}
    except:
        return {}
