from shutil import copyfile
import datetime
import csv
import os

def mkRunDir(env, config=None, sequenceid=None, runid=None):
    '''
    Method creates run folder, within which stats are stored.
    :param object: env(ironment)
    :return: csv writer opbject
    '''
    now = datetime.datetime.now()

  
    if sequenceid is None:
        folder = now.strftime('Results/' + env.name + '/%d-%b-%Y %H:%M:%S/')
    else:
        folder = now.strftime('Results/' + env.name + '/%d-%b-%Y/' +\
                              config.drl + "_" + str(config.madrl)  +\
                              '/' +str(sequenceid) + "_" + str(runid) +'_' + '%d-%b-%Y %H:%M:%S/')

    mkdir(folder)
    mkdir(folder + '/agent0')
    mkdir(folder + '/agent1')

    # CSV file is created for storing the results
    statscsv = folder + 'stats.csv'
    with open(statscsv, 'w') as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
         writer.writeheader()
         return statscsv, folder

def mkdir(path):
    '''
    Make stats folder
    :param string folder: folder to be created
    '''
    try: os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path): pass
        else: raise
