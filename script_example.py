# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    Script example, execute as:

    python script_example.py

'''
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import matplotlib.pyplot as plt
import numpy as np

# Local modules
import signal_plateau_recognition as spr

if __name__ == '__main__':

    # Some input data simulating 2 plateaus with oscillations
    xx = np.linspace(0, 4, 400)
    yy = 2 + 0.01*np.cos(100*xx)
    yy[(xx < 1) | (xx > 3.8)] = 0
    yy[(xx > 3) & (xx < 3.8)] = 1 + 0.01*np.cos(80*xx[(xx > 3) & (xx < 3.8)])

    # Find plateaus
    plat = spr.StatsPlateau(xx, yy)
    print('Initial time plateaus  =', plat.tIniPlateau)
    print('Final time plateaus    =', plat.tEndPlateau)
    print('Duration plateaus      =', plat.dtPlateau)
    print('Average value plateaus =', plat.plateauVal)

    # Plot detected plateaus
    plt.plot(xx, yy, label='signal', linewidth=2)
    for ii in range(len(plat._maskRefReq)):
        plt.plot(xx[plat._maskRefReq[ii]], yy[plat._maskRefReq[ii]], \
                 label='plateau ' + str(ii+1), linewidth=2)
    plt.legend()
    plt.show()
