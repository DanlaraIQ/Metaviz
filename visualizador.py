# Paquetes requeridos
import random
import numpy as np
import optimizadores as opt
import math
# from solution import solution
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import Problema as pro
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use('seaborn')

# - Funciones de prueba para visualización

# --- Suma de cuadrados
# --- Límites de [-100, -100] a [100, 100]


def f1(g):
    x = g[0]
    y = g[1]
    return x**2 + y**2

# --- SCHWEFEL_FUNCTION
# --- Límites de [-500, -500] a [500, 500]


def f2(g):
    x = g[0]
    y = g[1]
    return 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y))))


def metavisual(lb, ub, dim, numInd, numRun, optimizador, fobj):
    x_1 = np.linspace(lb[0], ub[0], 60)
    y_1 = np.linspace(lb[1], ub[1], 60)
    X_gra, Y_gra = np.meshgrid(x_1, y_1)
    Z_gra = np.zeros([60, 60])
    for i in range(60):
        for j in range(60):
            Z_gra[i, j] = fobj([x_1[i], y_1[j]])

    datos_grafica_movimiento, mejor_eval, mejor_ruta, gBest = optimizador(lb, ub, dim, numInd, numRun, fobj)

    datos_grafica_movimiento = np.asarray(datos_grafica_movimiento)
    cont = 0

    U = []
    V = []

    for i in range(numRun):
        plt.figure()
        ax = plt.gca()

        X = U
        Y = V
        U = []
        V = []

        for k in range(numInd):
            for l in range(dim):
                if cont == 0 or cont % 2 == 0:
                    U.append(datos_grafica_movimiento[cont])
                else:
                    V.append(datos_grafica_movimiento[cont])
                cont += 1

        if i > 0:

            XX = np.asarray(X)
            YY = np.asarray(Y)
            UU = np.asarray(U)
            VV = np.asarray(V)
            plt.contour(X_gra, Y_gra, Z_gra, 20, linewidths=1, cmap="viridis")
            ax.quiver(XX, YY, UU - XX, VV - YY, angles='xy', scale_units='xy', scale=1, width=0.005)
            ax.scatter(X, Y, color="r")
            # ax.scatter(U, V)
            ax.set_xlim([lb[0], ub[0]])
            ax.set_ylim([lb[1], ub[1]])
            plt.title("Iteración " + str(i) + " con " + str(numInd) + " individuos")
            plt.xlabel("$x_1$", fontsize=16)
            plt.ylabel("$x_2$", fontsize=16)
            plt.savefig(str(i) + ".png")

    nombre = 1
    filenames = []
    for i in range(numRun - 1):
        nom = str(nombre) + ".png"
        filenames.append(nom)
        nombre += 1

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('/Users/oscarlara/Dropbox/proyectosGIT/MetaVisual/scripts/animacion.gif', images)


fobj = f2
lb = np.array([-500, -500])
ub = np.array([500, 500])


dim = len(ub)
numInd = 8
numRun = 75
optimizador = opt.cs

metavisual(lb, ub, dim, numInd, numRun, optimizador, fobj)
