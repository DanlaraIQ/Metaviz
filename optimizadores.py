import random
import numpy as np
import math
# from solution import solution
import time

# -- PSO


def pso(lb, ub, dim, PopSize, iters, fobj):
    # PSO parameters
    mejor_ruta = []
    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    vel = np.zeros((PopSize, dim))
    pBestScore = np.zeros(PopSize)
    pBestScore.fill(float("inf"))
    pBest = np.zeros((PopSize, dim))
    gBest = np.zeros(dim)
    gBestScore = float("inf")
    datos_grafica_movimiento = []
    pos = np.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb
    tiempo_inicial = time.time()
    for l in range(0, iters):
        una_linea = pos.ravel()
        datos_grafica_movimiento.extend(una_linea)
        for i in range(0, PopSize):
            pos[i, :] = np.clip(pos[i, :], lb, ub)
            # Calculate objective function for each particle
            fitness = fobj(pos[i, :])

            if(pBestScore[i] > fitness):
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :]

            if(gBestScore > fitness):
                gBestScore = fitness
                gBest = pos[i, :]

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = w * vel[i, j] + c1 * r1 * \
                    (pBest[i, j] - pos[i, j]) + c2 * r2 * (gBest[j] - pos[i, j])

                if(vel[i, j] > Vmax):
                    vel[i, j] = Vmax

                if(vel[i, j] < -Vmax):
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]
        mejor_ruta.append(gBestScore)
    mejor_eval = gBestScore
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, mejor_eval, mejor_ruta, gBest

# - DE


def de(lb, ub, dim, PopSize, iters, fobj):
    valorMejor = []
    mutation_factor = 0.5
    crossover_ratio = 0.7
    stopping_func = None
    best = float('inf')

    # initialize population
    population = []
    datos_grafica_movimiento = []

    population_fitness = np.array([float("inf") for _ in range(PopSize)])

    for p in range(PopSize):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = np.array(population)

    # calculate fitness for all the population
    for i in range(PopSize):
        fitness = fobj(population[i, :])
        population_fitness[p] = fitness

        # is leader ?
        if fitness < best:
            best = fitness
            leader_solution = population[i, :]

    # start work

    t = 0
    tiempo_inicial = time.time()
    while t < iters:
        una_linea = population.ravel()
        datos_grafica_movimiento.extend(una_linea)
        # should i stop
        if stopping_func is not None and stopping_func(best, leader_solution, t):
            break

        # loop through population
        for i in range(PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in range(PopSize) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (
                    population[id_2, d] - population[id_3, d]
                )

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = np.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = fobj(mutant_sol)
            # s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < best:
                    best = mutant_fitness
                    leader_solution = mutant_sol

        # increase iterations
        t = t + 1

        valorMejor.append(best)

    # return solution
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, best, valorMejor, leader_solution

# - GWO


def gwo(lb, ub, dim, SearchAgents_no, Max_iter, fobj):
    valorMejor = []
    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")
    datos_grafica_movimiento = []

    # Initialize the positions of search agents
    Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb

    # Main loop
    tiempo_inicial = time.time()
    for l in range(0, Max_iter):
        una_linea = Positions.ravel()
        datos_grafica_movimiento.extend(una_linea)
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)

            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a  # Equation (3.3)
                C1 = 2 * r2  # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])  # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha  # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a  # Equation (3.3)
                C2 = 2 * r2  # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])  # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a  # Equation (3.3)
                C3 = 2 * r2  # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])  # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
        valorMejor.append(Alpha_score)
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, Alpha_score, valorMejor, Alpha_pos

    # - CS


def get_cuckoos(nest, best, lb, ub, n, dim):

    # perform Levy flights
    tempnest = np.zeros((n, dim))
    tempnest = np.array(nest)
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)

    s = np.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = np.random.randn(len(s)) * sigma
        v = np.random.randn(len(s))
        step = u / abs(v)**(1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * np.random.randn(len(s))

        for k in range(dim):
            tempnest[j, k] = np.clip(s[k], lb[k], ub[k])

    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf):
    # Evaluating all new solutions
    tempnest = np.zeros((n, dim))
    tempnest = np.copy(nest)

    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = objf(newnest[j, :])
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmin = min(fitness)
    K = np.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness
# Replace some nests by constructing new solutions/nests


def empty_nests(nest, pa, n, dim):

    # Discovered or not
    tempnest = np.zeros((n, dim))

    K = np.random.uniform(0, 1, (n, dim)) > pa

    stepsize = random.random() * (nest[np.random.permutation(n),
                                       :] - nest[np.random.permutation(n), :])

    tempnest = nest + stepsize * K

    return tempnest
##########################################################################


def cs(lb, ub, dim, n, N_IterTotal, objf):
    valorMejor = []
    datos_grafica_movimiento = []

    # Discovery rate of alien eggs/solutions
    pa = 0.25

    nd = dim
    # RInitialize nests randomely
    nest = np.zeros((n, dim))
    for i in range(dim):
        nest[:, i] = np.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]

    new_nest = np.zeros((n, dim))
    new_nest = np.copy(nest)

    bestnest = [0] * dim

    fitness = np.zeros(n)
    fitness.fill(float("inf"))

    fmin, bestnest, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)
    convergence = []
    # Main loop counter
    tiempo_inicial = time.time()
    for iter in range(0, N_IterTotal):
        una_linea = nest.ravel()
        datos_grafica_movimiento.extend(una_linea)
        # Generate new solutions (but keep the current best)

        new_nest = get_cuckoos(nest, bestnest, lb, ub, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        new_nest = empty_nests(new_nest, pa, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

        valorMejor.append(fmin)
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, fmin, valorMejor, bestnest

    # - TLBO


def tlbo(lb, ub, dim, ind, iter, fobj):
    valorMejor = []
    datos_grafica_movimiento = []
    # 1 Inicializando la población
    pos = lb + np.random.uniform(0, 1, size=(ind, dim)) * (ub - lb)
    pos_new = np.zeros((ind, dim))
    pos_eval = np.zeros(ind)
    mejor_pos = np.zeros(dim)
    mejor_pos_eval = float('inf')

    tiempo_inicial = time.time()
    for k in range(iter):
        una_linea = pos.ravel()
        datos_grafica_movimiento.extend(una_linea)
        for i in range(ind):
            pos_eval[i] = fobj(pos[i, :])

            if pos_eval[i] <= mejor_pos_eval:
                mejor_pos_eval = pos_eval[i]
                mejor_pos = pos[i, :]

        mean = np.mean(pos, axis=0)
        Tf = round(1 + random.random())
        for i in range(ind):
            pos_new[i, :] = pos[i, :] + np.random.uniform(0, 1, size=dim) * (mejor_pos - Tf * mean)
            pos_new[i, :] = np.clip(pos_new[i, :], lb, ub)

            pos_new_eval = fobj(pos_new[i, :])

            if pos_new_eval < pos_eval[i]:
                pos_eval[i] = pos_new_eval
                pos[i, :] = pos_new[i, :]

        for i in range(ind):
            index1 = random.randint(0, ind - 1)
            if index1 == i:
                while index1 == i:
                    index1 = random.randint(0, ind - 1)

            eval_p = fobj(pos[index1, :])
            if pos_eval[i] < eval_p:
                pos_new[i, :] = pos[i, :] + \
                    np.random.uniform(0, 1, size=dim) * (pos[i, :] - pos[index1, :])

            else:
                pos_new[i, :] = pos[i, :] + np.random.uniform(0, 1) * (pos[index1, :] - pos[i, :])

            pos_new[i, :] = np.clip(pos[i, :], lb, ub)

            eval_n = fobj(pos_new[i, :])

            if eval_n < pos_eval[i]:
                pos_eval[i] = eval_n
                pos[i, :] = pos_new[i, :]
        valorMejor.append(mejor_pos_eval)
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, mejor_pos_eval, valorMejor, mejor_pos
