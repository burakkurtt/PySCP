"""
author: HBKurt
date: 2023-01-11
"""
import matplotlib.pyplot as plt
import numpy as np
from lib.utils.plotlib import *
from lib.core.Entrance import PySCP
from lib.utils.Scale import Scale
from lib.utils.Structs import PhaseInfo


class Brachistochrone:
    """
        Brachistocrone Problem
    """
    def __init__(self):
        # scale
        scale = Scale(state_name=['non', 'non', 'non'], control_name=['non'])
        phase = PhaseInfo()
        phase.init_state_bound = np.array([[0., 0., 0.], [0., 0., 0.]])
        phase.state_bound = np.array([[0., 0., 0.], [2., 2., 10.]])
        phase.final_state_bound = np.array([[2., 2., 0.], [2., 2., 0.]])
        phase.control_bound = np.array([[-10.], [10.]])
        phase.t0_bound = np.array([0., 0.])
        phase.tf_bound = np.array([1., 1.])
        phase.dynamicsFunc = self.Dynamics
        phase.pathFunc = self.Path
        phase.scale = scale
        phase.trState = np.ones((3,)) * 1 # tr : Thrust Region
        phase.trControl = np.ones((1,)) * 1
        phase.trSigma = 1
        self.phases = [phase]
        self.linkages = []
        self.phaseNum = len(self.phases)

    def Dynamics(self, state, control, t, auxdata):
        x, y, v = state
        u = control[0]

        g = 9.81

        # original dynamic equation
        f = np.zeros_like(state)
        f[0] = v * np.sin(u)
        f[1] = v * np.cos(u)
        f[2] = g * np.cos(u)

        # approximated dynamic equation
        A = np.array([[0, 0, np.sin(u)], [0, 0, np.cos(u)], [0, 0, 0]])
        B = np.array([[v * np.cos(u)], [-1 * v * np.sin(u)], [-1 * g * np.sin(u)]])

        return f, A, B

    def Path(self, phases):
        path = None
        return [path]

    def objective(self, Vars, refTraj, weights4State, weights4Control):
        obj = weights4Control[0] @ (refTraj[0].sigma * (Vars[0].control[:, 0] ** 2)) / 2.
        return obj


def singleCase(setup):
    prob = PySCP(setup)
    prob.solve()
    error = prob.result.maxError
    cputime = prob.result.cvxTime
    obj = prob.result.objective
    return obj, cputime, error


def multipleCases():
    ssList = [10, 20, 50, 100]
    hpList = [[1, 1, 5, 5],
              [20, 40, 10, 20]]
    num = len(ssList)

    cpuTimes = np.zeros((num,))
    objs = np.zeros((num,))
    errors = np.zeros((num, 2))

    vehicle = Brachistochrone()
    for i in range(num):
        PSConfig = {'meshName': 'LG',
                    'scheme': 'integral',  # difference or integration
                    'adaptive': False,
                    'degree': [hpList[1][i]] * hpList[0][i],
                    'segFractions': [1. / hpList[0][i]] * hpList[0][i]}
        SSConfig = {'meshName': 'RK',
                    'adaptive': False,
                    'ncp': ssList[i]}
        setup = {'model': vehicle,
                 'meshConfig': PSConfig,
                 'algorithm': 'TrustRegion',
                 'adaptive': False,
                 'initialization': 'linear',
                 'maxIteration': 10,
                 'verbose': 0}
        obj, cputime, error = singleCase(setup)
        cpuTimes[i] = cputime
        objs[i] = obj
        errors[i] = error[0]

    print('目标：', objs)
    print('目标相对误差', np.abs(np.array(objs) - 40 / 9) / (40 / 9))
    print('用时：', cpuTimes)
    print('误差：', errors)


def compare():
    ssList = 10 * np.arange(1, 11)
    ppList = 10 * np.arange(1, 11)
    hhList = np.arange(1, 11)
    num = len(ssList)

    cpuTimes = np.zeros((num,))
    objs = np.zeros((num,))
    errors = np.zeros((num, 2))
    markers = 'xos^*<^*<'

    plt.figure()
    vehicle = Brachistochrone()
    labels = ['ZOH', 'FOH', 'RK', 'p-LG', 'p-LGR', 'p-LGR', 'h-LG', 'h-LGR', 'h-fLGR']

    for imethod in range(9):
        for i in range(num):
            method1 = {'meshName': 'ZOH', 'ncp': ssList[i]}
            method2 = {'meshName': 'FOH', 'ncp': ssList[i]}
            method3 = {'meshName': 'RK', 'ncp': ssList[i]}
            method4 = {'meshName': 'LG', 'scheme': 'integral', 'degree': [ppList[i]] * 1, 'segFractions': [1. / 1] * 1}
            method5 = {'meshName': 'LGR', 'scheme': 'integral', 'degree': [ppList[i]] * 1, 'segFractions': [1. / 1] * 1}
            method6 = {'meshName': 'fLGR', 'scheme': 'integral', 'degree': [ppList[i]] * 1, 'segFractions': [1. / 1] * 1}
            method7 = {'meshName': 'LG', 'scheme': 'integral', 'degree': [10] * hhList[i], 'segFractions': [1. / hhList[i]] * hhList[i]}
            method8 = {'meshName': 'LGR', 'scheme': 'integral', 'degree': [10] * hhList[i], 'segFractions': [1. / hhList[i]] * hhList[i]}
            method9 = {'meshName': 'fLGR', 'scheme': 'integral', 'degree': [10] * hhList[i], 'segFractions': [1. / hhList[i]] * hhList[i]}
            methods = [method1, method2, method3, method4, method5, method6, method7, method8, method9]
            setup = {'model': vehicle,
                     'meshConfig': methods[imethod],
                     'maxIteration': 15,
                     'verbose': 0}
            obj, cputime, error = singleCase(setup)
            cpuTimes[i] = cputime
            objs[i] = obj
            errors[i] = error[0]

        print('目标：', objs)
        bestObj = 40 / 9
        objError = (np.abs(np.array(objs) - bestObj) / bestObj)
        print('目标相对误差', objError)
        print('用时：', cpuTimes)
        print('误差：', errors)

        plt.figure(1)
        plt.plot(ssList, np.max(errors, axis=1), marker=markers[imethod], label=labels[imethod])

        plt.figure(2)
        plt.plot(ssList, objError, marker=markers[imethod], label=labels[imethod])

    plt.figure(1, figsize=(8, 6))
    plt.xticks(ssList)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Node Number')
    plt.ylabel('State Error')
    plt.tight_layout()
    plt.savefig('Integration Error.eps', dpi=600)

    plt.figure(2, figsize=(8, 6))
    plt.yscale('log')
    plt.xticks(ssList)
    plt.legend()
    plt.xlabel('Node Number')
    plt.ylabel('Objective Error')
    plt.tight_layout()
    plt.savefig('Objective Error.eps', dpi=600)
    plt.show()


def paper():
    vehicle = Brachistochrone()

    PSConfig = {'meshName': 'LGR', 'scheme': 'integral', 'degree': [40] * 2, 'segFractions': [0.5] * 2}
    SSConfig = {'meshName': 'RK', 'ncp': 40}

    setup = {'model': vehicle,
             'meshConfig': PSConfig,
             'maxIteration': 1,
             'weightNu': [100],
             'plot_interval': 100,
             'verbose': 2}

    prob = PySCP(setup)
    # prob.plotXU(traj=prob.result.solutionDimension, show=False)
    prob.solve()
    prob.plotXU(traj=prob.result.solutionIntegrated, marker='.')
    prob.plotXU(traj=prob.result.solutionDimension,
                show=True,
                save=False,
                # matlab_path='breakwell.mat',
                state_name=[r'$x$', r'$y$', r'$v$'],
                control_name=['u'],
                legend=['Initial Guess', 'Open Loop', 'PySCP'])
    prob.print()


if __name__ == '__main__':
    paper()
    # multipleCases()
    # compare()
