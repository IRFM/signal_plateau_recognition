# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    MULTI SHOT RUN FOR SPECIFIED FUNCTION
'''
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import argparse
from concurrent import futures
import glob
import numpy as np
import os
import pandas as pd
import re
import scipy.io
import subprocess
import sys
import warnings

#print('path 1 =', sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#print('path 2 =', sys.path)

mem_limit_shots = 20 # Max numbers of shots to avoid memory limit


def multi_shot(function=None, module=None, shot_list=None, shot_init_end=None, \
               filepath='./', run=0, occ=0, user='imas_public', machine='west', \
               procs_max=5, dry_run=False):

    print(' ')
    print('In function mult_shot')
    print(' ')
    print('function   =', function)
    print('module     =', module)
    if (shot_list != None):
        print('shot_list  =', shot_list)
    elif (shot_init_end != None):
        print('shot_init  =', shot_init_end[0])
        print('shot_end   =', shot_init_end[1])
    print('filepath   =', filepath)
    print('run        =', run)
    print('occurrence =', occ)
    print('user       =', user)
    print('machine    =', machine)
    print(' ')

    # Import module
    exec('import ' + str(module))

    print('IMAS_VERSION batch =', os.environ['IMAS_VERSION'])
    print(' ')

    if (shot_init_end != None):
        path_files = os.path.expanduser('~' + user + '/public/imasdb/' + \
                                        machine + '/3/0/ids_*.datafile')
        print('path_files batch =', path_files)

        shot_files = glob.glob(path_files)
        #print('shot_files batch =', shot_files)
        print('len(shot_files) batch =', len(shot_files))
        try:
            shot_files[0]
        except:
            print(' ')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('ERROR: no files for requested shot')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(' ')
            raise

        int_shot = []

        count = 0
        for ifile in shot_files:
            ids_file = ifile.split('/')
            number_file = re.findall('\d', ids_file[-1])

            if len(number_file) <= 4:
               continue

            number_shot = number_file[:-4]

            int_shot.append(int(''.join(number_shot)))

            count += 1

        set_shot = set(int_shot)
        print(' ')

        # Make NumPy array to be able to use mask
        sorted_shot = np.asarray(sorted(set_shot))

        print(' ')
        print('First shot =', shot_init_end[0])
        print('Last shot  =', shot_init_end[1])
        print(' ')

        mask = (sorted_shot >= shot_init_end[0]) & \
               (sorted_shot <= shot_init_end[1])

        shot_list = sorted_shot[mask]

    print('')
    print('LAUNCH PARALLEL JOBS')
    print('--------------------')
    print('')

    max_workers_input = min(len(shot_list), procs_max)
    print('len(shot_list)    =', len(shot_list))
    print('max_workers_input =', max_workers_input)
    print('')

    if (len(shot_list) > mem_limit_shots):
        sections = np.int(np.ceil(shot_list.size / mem_limit_shots))
        print('Shot list larger than mem_limit_shots, sections =', sections)
        print('')
        sec_shots = np.array_split(shot_list, sections)
    else:
        sec_shots = [shot_list]

    print('length of each sec_shots =', [len(sec) for sec in sec_shots])
    print('')

    if (not dry_run):

        count = 0

        for isec_shots in sec_shots:
            with futures.ProcessPoolExecutor(max_workers=max_workers_input) \
                as executor:
                # Disable print
                #sys.stdout = open(os.devnull, 'w')

                # Dictionary comprehension, this allows to retrieve
                # the corresponding shot
                # from the future (dic_ftr) dictionary obtained
                future_fct = executor.map(eval(str(module) + '.' + str(function)), \
                                        [int(iishot) for iishot in isec_shots], \
                                        chunksize=len(isec_shots)//max_workers_input)

                out = list(future_fct)

                lshot     = [None]*len(isec_shots)
                out_stats = [None]*len(isec_shots)
                out_sig   = [None]*len(isec_shots)
                out_info  = [None]*len(isec_shots)
                out_act   = [None]*len(isec_shots)
                for ii in range(len(out)):
                    lshot[ii]     = str(out[ii][2]['shot'])
                    out_stats[ii] = out[ii][0]
                    out_sig[ii]   = out[ii][1]
                    out_info[ii]  = out[ii][2]
                    out_act[ii]   = \
                      {(jj, kk): out[ii][3][jj][kk] for jj in out[ii][3] \
                                 for kk in range(len(out[ii][3][jj]))}
                    out_act[ii] = pd.DataFrame.from_dict(out_act[ii], orient='index')
                    if (len(out_act[ii]) != 0):
                        out_act[ii].columns = [lshot[ii]]
                        out_act[ii].index = pd.MultiIndex.from_tuples(out_act[ii].index)

            dict_out_stats = dict(zip(lshot, map(pd.DataFrame.from_dict, out_stats)))
            dataStats = pd.concat(dict_out_stats.values(), keys=dict_out_stats.keys())
            dataSig   = pd.DataFrame.from_dict(dict(zip(lshot, out_sig))).transpose()
            dataInfo  = pd.DataFrame.from_dict(dict(zip(lshot, out_info))).transpose()
            dataAct   = pd.concat(out_act, axis=1).transpose()

            try:
                os.remove( \
         '{0}data_IMAS_stats_Shots_{1}_{2}_Run{3:04d}_Occ{4}_{5}_{6}_{7}.h'.format( \
         filepath, shot_list[0], shot_list[-1], run, occ, user, machine, count))
            except FileNotFoundError as err:
                warnings.warn('Files not found'+str(err))

            store = pd.HDFStore( \
         '{0}data_IMAS_stats_Shots_{1}_{2}_Run{3:04d}_Occ{4}_{5}_{6}_{7}.h'.format( \
         filepath, shot_list[0], shot_list[-1], run, occ, user, machine, count))

            store['info']      = dataInfo
            store['stats']     = dataStats
            store['signals']   = dataSig
            store['actuators'] = dataAct
            store.close()
            count += 1

        info_in      = [None]*len(sec_shots)
        stats_in     = [None]*len(sec_shots)
        signals_in   = [None]*len(sec_shots)
        actuators_in = [None]*len(sec_shots)

        for ii in range(len(sec_shots)):
            store = pd.HDFStore( \
         '{0}data_IMAS_stats_Shots_{1}_{2}_Run{3:04d}_Occ{4}_{5}_{6}_{7}.h'.format( \
         filepath, shot_list[0], shot_list[-1], run, occ, user, machine, ii))

            info_in[ii]      = store['info']
            stats_in[ii]     = store['stats']
            signals_in[ii]   = store['signals']
            actuators_in[ii] = store['actuators']

            store.close()

        info_all      = pd.concat(info_in, axis=0)
        stats_all     = pd.concat(stats_in, axis=0)
        signals_all   = pd.concat(signals_in, axis=0)
        actuators_all = pd.concat(actuators_in, axis=0)

        store = pd.HDFStore( \
        '{0}data_IMAS_stats_Shots_{1}_{2}_Run{3:04d}_Occ{4}_{5}_{6}.h'.format( \
        filepath, shot_list[0], shot_list[-1], run, occ, user, machine))

        store['info']      = info_all
        store['stats']     = stats_all
        store['signals']   = signals_all
        store['actuators'] = actuators_all

        store.close()

        for ii in range(len(sec_shots)):
            try:
                os.remove( \
         '{0}data_IMAS_stats_Shots_{1}_{2}_Run{3:04d}_Occ{4}_{5}_{6}_{7}.h'.format( \
         filepath, shot_list[0], shot_list[-1], run, occ, user, machine, ii))
            except FileNotFoundError as err:
                print('Files not found', err)
    else:
        print(' ')
        print('WARNING: DRY-RUN')
        print(' ')

if __name__ == '__main__':

    # Parse input arguments
    parser = argparse.ArgumentParser(description= \
            '''Script reading IMAS database, launching function
               for shot interval and storing HDF file at filepath''')
    parser.add_argument('function', type=str, \
                        help='function to call for each shot')
    parser.add_argument('module', type=str, \
                        help='function module')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--shot-list', type=int, \
                       nargs='+', \
                       help='option for manual shot list')
    group.add_argument('-ie', '--shot-init-end', type=int, \
                       nargs=2, default=None, \
                       help='option for manual interval shot init and shot end')
    parser.add_argument('--filepath', type=str, default='./', \
                        help='filepath to store statistics results, default: current path')
    parser.add_argument('--procs_max', type=int, default=5, \
                        help='procs_max, default=5')
    parser.add_argument('--run', type=int, default=0, \
                        help='run number, default=0')
    parser.add_argument('--occ', type=int, default=0, \
                        help='occurrence number, default=0')
    parser.add_argument('--user', type=str, default='imas_public', \
                        help='user, default=imas_public')
    parser.add_argument('--machine', type=str, default='west', \
                        help='machine, default=west')
    parser.add_argument('--dry-run', action='store_true', \
                        help='DRY RUN')

    args = parser.parse_args()

    multi_shot(args.function, args.module, args.shot_list, args.shot_init_end, \
               args.filepath, args.run, args.occ, args.user, args.machine, \
               args.procs_max, args.dry_run)
