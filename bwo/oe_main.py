from random import uniform
from random import choice
from random import random
from random import randint
from copy import deepcopy

from landscapes.single_objective import sphere
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import statistics as st
import pandas as pd

logging.basicConfig(filename="oe_main.log")

import sys

sys.tracebacklimit = 0


def _generate_new_position(x0: list = None, dof: int = None, bounds: list = None) -> list:
    '''GENERATE NEW POSITION

    Parameters
    ----------
    dof : int
    x0 : list
    bounds : list of tuples [(x1_min, x1_max),...,(xn_min, xn_max)]

    Returns
    -------
    list

    Notes
    -----
    There are several ways in which an initial position can be generated.
    Outlined below are all possible scenarios and outputs.

    nomenclature:
        "dof" = "degrees of freedom" = "dimensions" = "d"
        p = new initial position vector of length d

    just bounds:
        for each position i in bounds,  p[i] = random value in [i_min, i_max]]

    just x0:
        for each position i in x0: p[i] = x0[i] + random value in [-1, 1]

    just dof:
        for each position i from 0 to d,  p[i] = random value in [-1, 1]

    dof + x0:
        since dof and x0 are redundent from a dimensionality perspective,
        this situation will defer to the case above "just x0".

    dof + bounds:
        since dof and bounds are redundent from a dimensionality perspective,
        this situation wll defer to the case above "just bounds"

    x0 + bounds:
        for each position i in x0:
            p[i] = x0[i] + random value in [-1, 1] constrained by bounds[i].min
            and bounds[i].max

    dof + x0 + bounds:
        see case: "x0 + bounds" above

    All this boils down to four cases (ordered by information gain from user):
    1) x0 and bounds
    2) bounds
    3) x0
    4) dof
    '''

    if x0 and bounds:
        return [min(max(uniform(-1, 1) + x0[i], bounds[i][0]), bounds[i][1]) for i in range(len(x0))]

    if bounds:
        return [uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

    if x0:
        return [x_i + uniform(-1, 1) for x_i in x0]

    if dof:
        return [uniform(-1, 1) for _ in range(0, dof)]


def minimize(func, x0=None, dof=None, bounds=None, pp=0.6, cr=0.44, pm=0.4, npop=10, disp=False, maxiter=50, rmax=2,
             rmin=1.5, dynamic=False):
    '''
    Parameters
    ----------
    x0 : list
    	initial guess
    pp : float
    	procreating percentage
    cr : float
    	cannibalism rate. A cr of 1 results in all children surviving. A cr of 0
        results in no children surviving
    pm : float
        mutation rate

    Returns
    -------
    float : solution at global best
    list : position at global best

    References
    ----------
    '''

    # do some basic checks before going any further
    assert type(disp) == bool, 'parameter: disp -> must be of type: bool'
    assert type(npop) == int, 'parameter: npop -> must be of type: int'
    assert type(maxiter) == int, 'parameter: maxiter -> must be of type int'
    if x0 is not None: assert type(x0) == list, 'x0 must be of type: list'
    if dof is not None: assert type(dof) == int, 'parameter: dof -> must be of type: int'
    if bounds is not None: assert type(bounds) == list, 'parameter: bounds -> must be of type list'
    assert x0 is not None or dof is not None or bounds is not None, 'must specify at least one of the following: x0, dof, or bounds'
    if x0 and bounds: assert len(bounds) == len(x0), 'x0 and bounds must have same number of elements'
    assert pp > 0 and pp <= 1, 'procreating percentage "pp" must be: 0 < pp <= 1'
    assert cr >= 0 and cr <= 1, 'cannibalism rate "cr" must be: 0 < cr <= 1'
    assert pm >= 0 and pm <= 1, 'mutation rate "pm" must be: 0 < pm <= 1'
    assert maxiter > 0, 'maxiter must be greater than zero.'

    # check bounds specification if necessary
    if bounds:
        assert type(bounds) == list, 'bounds must be of type: list'
        for b in bounds:
            assert type(
                b) == tuple, 'element in bounds is not of type: tuple. ever every element must be a tuple as specified (v_min, v_max)'
            assert b[0] < b[1], 'element in bounds specified incorrectly. must be (xi_min, xi_max)'

    # constants
    if x0 is not None:
        dof = len(x0)
    elif bounds is not None:
        dof = len(bounds)

    nr = int(npop * pp)  # number of reproduction
    nm = int(npop * pm)  # number of mutation children
    spacer = len(str(npop))  # for logging only

    # initialize population
    pop = [_generate_new_position(x0, dof, bounds) for _ in range(0, npop)]

    # main loop
    hist = []
    total_pop = []
    total_pp = []
    total_pm = []
    cont = 0

    for epoch in range(0, maxiter):

        # initialize epoch
        # New: Población dinámica
        largo_pop = len(pop)
        pop = sorted(pop, key=lambda x: func(x), reverse=False)
        if (dynamic):
            var1 = int(npop * rmax)
            var2 = int(npop * rmin)
            var3 = int(pow(npop, 2))
            nr = int(largo_pop * pp)
            if largo_pop > var1 and cont == 0:
                # Disminuir la población
                cr = 1 - cr
                pm = 1 - pm
                pp = 1 - pp
                cont = 1
            elif largo_pop < var2 and cont == 1:
                # Aumentar la población
                cr = 1 - cr
                pm = 1 - pm
                pp = 1 - pp
                cont = 0
            if largo_pop > var3:
                pop = pop[:npop]
                nr = len(pop)
        # End Población Dinámica

        pop1 = deepcopy(pop[:nr])
        pop2 = []
        pop3 = []
        gbest = pop[0]

        # print something useful
        if disp: print(f'> ITER: {epoch + 1:>{spacer}} | GBEST: {func(gbest):0.9f}')

        # procreation and cannibalism
        for i in range(0, nr):

            # randomly pick two parents
            i1, i2 = randint(0, len(pop1) - 1), randint(0, len(pop1) - 1)
            p1, p2 = pop1[i1], pop1[i2]

            # crossover
            children = []
            # New: Evitar que la población sea cero
            for j in range(0, dof):
                # generate two new children using equation (1)
                alpha = random()
                c1 = [(alpha * v1) + ((1 - alpha) * v2) for v1, v2 in zip(p1, p2)]
                c2 = [(alpha * v2) + ((1 - alpha) * v1) for v1, v2 in zip(p1, p2)]

                # persist new children to temp population
                children.append(c1)
                children.append(c2)

            # cannibalism - destroy male; since female black widow spiders are
            # larger and often end up killing the male during mating, we'll
            # assume that the fitter partent is the female. thus, we'll delete
            # the weaker parent.
            if func(p1) > func(p2):
                pop1.pop(i1)
            else:
                pop1.pop(i2)

            # cannibalism - destroy some children
            children = sorted(children, key=lambda x: func(x), reverse=False)

            # New: Modificar la cantidad de hijos que sobreviven
            total_children = int(len(children))
            indice = int(total_children - max((total_children * cr), 1))

            children = children[:indice]

            # add surviving children to pop2
            pop2.extend(children)

        # mutation

        # New: Modificar nm para que ocupe la cantidad de pop2
        nm = int(len(pop2) * pm)  # number of mutation children
        for i in range(0, nm):
            # pick a random child
            # m = choice(pop2)

            # New: Eliminar de pop2 el hijo seleccionado para la mutación
            index_pop2 = randint(0, len(pop2) - 1)
            m = pop2[index_pop2]

            pop2.pop(index_pop2)

            # pick two random chromosome positions
            cp1, cp2 = randint(0, dof - 1), randint(0, dof - 1)

            # New: Validar que los índices sean diferentes
            while cp1 == cp2:
                cp1, cp2 = randint(0, dof - 1), randint(0, dof - 1)

            # swap chromosomes
            m[cp1], m[cp2] = m[cp2], m[cp1]

            # persist
            pop3.append(m)

        # assemble final population
        pop2.extend(pop3)
        pop = deepcopy(pop2)

        # return global best position and func value at global best position
        # print (f'epoch {epoch} - gbest { func(gbest)}')

        # Save epoch and fitness
        hist.append(func(gbest))
        total_pop.append(largo_pop)
        total_pp.append(pp)
        total_pm.append(pm)

    return func(gbest), gbest, hist, total_pop, total_pp, total_pm


def experimentos(n=31, npop=10, maxiter=30, rmax=2, rmin=0.8, dof=5, pp=0.6, pm=0.4, cr=0.4):
    data_frame_arr = []
    experimentos = []
    for i in range(n):
        start = datetime.now()

        fbest, xbest, hist, tpop, tpp, tpm = minimize(sphere, dof=dof, maxiter=maxiter, pp=pp, pm=pm, cr=cr, npop=npop,
                                                      rmax=rmax,
                                                      rmin=rmin, dynamic=True)
        experimentos.append(fbest)

        end = datetime.now()
        print(f'Experimento: {i + 1} de {n} - Duración: {end - start} - [{fbest}]')

        for x in range(0, len(hist)):
            data_frame_arr.append(
                [start, i + 1, x, hist[x], tpop[x], end, end - start, npop, maxiter, rmax, rmin, dof, tpp, tpm])

    df = pd.DataFrame(data_frame_arr,
                      columns=['start', 'experiment', 'epoch', 'fitness', 'totalpop', 'end', 'duration', 'npop',
                               'maxiter', 'rmax', 'rmin', 'dof', 'tpp', 'tpm'])

    df.to_csv('bwo.csv')

    return experimentos


if __name__ == "__main__":
    init_time = datetime.now()
    print(f'Inicio: {init_time}\n')

    n = 10
    npop = 100
    maxiter = 5
    rmax = 3
    rmin = 1
    dof = 2
    pp = 0.6
    pm = 0.4
    cr = 0.44

    data = experimentos(n, npop, maxiter, rmax, rmin, dof, pp, pm, cr)
    end_time = datetime.now()

    print(f'\nFin: {end_time}')

    log_text = f"""
        \nPara n={n}, npop={npop}, maxiter={maxiter}, rmax={rmax}, rmin={rmin}, dof={dof}:\n
        \tfbest={min(data)}
        \tfmean={st.mean(data)}
        \tfmedian={st.median(data)}
        \tfworst={max(data)}
        \tfstdev={st.stdev(data)}
        \nInicio: {init_time} - Fin: {end_time} - Duración: {(end_time - init_time)}
    """

    print(log_text)
    logging.warning(log_text)

    # Graficar experimentos
    plt.plot(data)
    plt.semilogy()
    plt.show()
