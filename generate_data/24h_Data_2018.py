# %load data_mafe_EI.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from obspy import UTCDateTime, read, Trace, Stream
from scipy import signal
import pandas as pd
from haversine import haversine

sta_data = pd.read_excel("/sdd1/sta_list.xlsx", sheet_name="Data")
sta_num = len(sta_data['latitude'])


def extract_seed(st_ex, time_DATE, save_dir, i_sta, i_cha, i_date_e, i_time_e, extract_time, time_DATE_E, event_i, ENZ):
    extract_data = [0]
    sample_rate = 0
    for st_list in st_ex:
        if (st_list.stats.starttime < time_DATE and st_list.stats.endtime > time_DATE):
            tr = st_list.copy()
            sample_rate = tr.stats.sampling_rate
            start_c = int((time_DATE - UTCDateTime(tr.stats.starttime)-5) * sample_rate)
            end_c = int(start_c + sample_rate * (extract_time + 60))
            extract_data = tr.data[start_c:end_c]

    latitude = 0.0
    longitude = 0.0
    depth = 0.0
    sta_i = 0
    while True:
        if sta_data['station_code'][sta_i] == i_sta or sta_data['code_old'][sta_i] == i_sta:
            latitude = sta_data['latitude'][sta_i]
            longitude = sta_data['longitude'][sta_i]
            depth = sta_data['height'][sta_i]
            break
        elif sta_i == sta_num - 1:
            print(i_sta)
            break
        sta_i += 1

    return extract_data, latitude, longitude, depth, sample_rate


def mafe_EI(fname, event_index, upper_dir, odata_dir, idx):
    t = 0
    extract_time = 10
    f = open(fname, 'r')

    if event_index == 0:
        save_dir = upper_dir + '/0'
    elif event_index == 1:
        save_dir = upper_dir + '/1'
    elif event_index == 2:
        save_dir = upper_dir + '/2'
    else:
        save_dir = upper_dir + '/3'

    while True:
        t = t + 1
        if t == 3:
            lines = f.readline().split()
            info_DATE_E = lines[2]
            info_TIME_E = lines[3]
            info_TIME_T = info_TIME_E[0:2] + '_' + info_TIME_E[3:5] + '_' + info_TIME_E[7:]
        elif t == 5:
            lines = f.readline().split()
            info_LAT = float(lines[2])

            lines = f.readline().split()
            info_LONG = float(lines[2])

            lines = f.readline().split()
            info_DEPTH = float(lines[2])

            lines = f.readline().split()
            info_MAG = float(lines[2])

        elif t >= 15:
            lines = f.readline().split()
            if len(lines) == 0 or lines[0] == 'First':
                break
            else:
                if info_MAG > 0.0:
                    info_STA = lines[0]
                    info_CHA = lines[1]
                    info_DATE = lines[3]
                    info_TIME = lines[4]

                    if len(lines) > 5:
                        if lines[5] == '/' and lines[6] == 'ml':
                            ### Event Time
                            UTC_DATE_E = info_DATE_E + 'T' + info_TIME_E
                            time_DATE_E = UTCDateTime(UTC_DATE_E)
                            DATA_julday_E = time_DATE_E.julday

                            ### Station Time
                            UTC_DATE = info_DATE + 'T' + info_TIME
                            time_DATE = UTCDateTime(UTC_DATE)
                            DATA_julday = time_DATE.julday

                            ### mseed extract
                            myjulday_name = '%03d' % (DATA_julday)
                            mydata_path = os.path.join(odata_dir, myjulday_name)
                            if info_CHA[0:2] == 'HH' or info_CHA[0:2] == 'EL':
                                info_CHA_E_HG = 'HGE'
                                info_CHA_N_HG = 'HGN'
                                info_CHA_Z_HG = 'HGZ'

                                myfile_name_E = '*.' + '%s.' % (info_STA) + '%s.' % (info_CHA_E_HG) + '*.*'
                                myfile_name_N = '*.' + '%s.' % (info_STA) + '%s.' % (info_CHA_N_HG) + '*.*'
                                myfile_name_Z = '*.' + '%s.' % (info_STA) + '%s.' % (info_CHA_Z_HG) + '*.*'

                                ext_file_E = glob.glob(os.path.join(mydata_path, myfile_name_E))
                                ext_file_N = glob.glob(os.path.join(mydata_path, myfile_name_N))
                                ext_file_Z = glob.glob(os.path.join(mydata_path, myfile_name_Z))

                                info_MAG_D = float(lines[7])
                                info_LABEL = event_index

                                event_info = {'event_DATE': info_DATE_E,
                                              'event_TIME': info_TIME_E,
                                              'event_LAT': info_LAT,
                                              'event_LONG': info_LONG,
                                              'event_DEPTH': info_DEPTH,
                                              'STA': info_STA,
                                              'CHA': info_CHA_E_HG,
                                              'station_DATE': info_DATE,
                                              'station_TIME': info_TIME,
                                              'MAG': info_MAG,
                                              'MAG_D': info_MAG_D,
                                              'LABEL_E': info_LABEL,
                                              'LABEL_W': 'none',
                                              'LABEL_D': 'none'
                                              }

                                if ext_file_E != [] and ext_file_N != [] and ext_file_Z != []:
                                    st_E = read(ext_file_E[0])  ## file reading
                                    st_N = read(ext_file_N[0])  ## file reading
                                    st_Z = read(ext_file_Z[0])  ## file reading

                                    tr_E, sta_LAT, sta_LONG, sta_DEPTH, sample_rate = extract_seed(st_E, time_DATE_E,
                                                                                                   save_dir, info_STA,
                                                                                                   info_CHA_E_HG,
                                                                                                   info_DATE_E,
                                                                                                   info_TIME_T,
                                                                                                   extract_time,
                                                                                                   time_DATE_E,
                                                                                                   event_info, 'E')
                                    tr_N, _, _, _, _ = extract_seed(st_N, time_DATE_E, save_dir, info_STA,
                                                                    info_CHA_N_HG, info_DATE_E, info_TIME_T,
                                                                    extract_time, time_DATE_E, event_info, 'N')
                                    tr_Z, _, _, _, _ = extract_seed(st_Z, time_DATE_E, save_dir, info_STA,
                                                                    info_CHA_Z_HG, info_DATE_E, info_TIME_T,
                                                                    extract_time, time_DATE_E, event_info, 'Z')

                                    if sta_LAT == 0.0 or sta_LONG == 0.0:
                                        continue
                                    if sample_rate <= 20:
                                        continue

                                    if len(tr_E) == 7000 and len(tr_N) == 7000 and len(tr_Z) == 7000:
                                        event_info['tr_E'] = tr_E
                                        event_info['tr_N'] = tr_N
                                        event_info['tr_Z'] = tr_Z
                                        event_info['sta_LAT'] = sta_LAT
                                        event_info['sta_LONG'] = sta_LONG
                                        event_info['sta_DEPTH'] = sta_DEPTH

                                        name_seed = event_info['STA'] + '_' + event_info['CHA']  ### mseed filename
                                        event_save_dir = save_dir + '/event/' + '{0:03}'.format(idx) + '/'
                                        if not os.path.isdir(event_save_dir):
                                            os.makedirs(event_save_dir)
                                        gen_name1 = event_save_dir + name_seed + '.npz'

                                        np.savez(gen_name1, **event_info)

                            if info_CHA[2] == 'E':
                                info_CHA_E = info_CHA
                                info_CHA_N = info_CHA[0:2] + 'N'
                                info_CHA_Z = info_CHA[0:2] + 'Z'


                            elif info_CHA[2] == 'N':
                                info_CHA_N = info_CHA
                                info_CHA_E = info_CHA[0:2] + 'E'
                                info_CHA_Z = info_CHA[0:2] + 'Z'

                            else:
                                info_CHA_Z = info_CHA
                                info_CHA_E = info_CHA[0:2] + 'E'
                                info_CHA_N = info_CHA[0:2] + 'N'

                            myfile_name_E = '*.' + '%s.' % (info_STA) + '%s.' % (info_CHA_E) + '*.*'
                            myfile_name_N = '*.' + '%s.' % (info_STA) + '%s.' % (info_CHA_N) + '*.*'
                            myfile_name_Z = '*.' + '%s.' % (info_STA) + '%s.' % (info_CHA_Z) + '*.*'

                            ext_file_E = glob.glob(os.path.join(mydata_path, myfile_name_E))
                            ext_file_N = glob.glob(os.path.join(mydata_path, myfile_name_N))
                            ext_file_Z = glob.glob(os.path.join(mydata_path, myfile_name_Z))

                            info_MAG_D = float(lines[7])
                            info_LABEL = event_index

                            event_info = {'event_DATE': info_DATE_E,
                                          'event_TIME': info_TIME_E,
                                          'event_LAT': info_LAT,
                                          'event_LONG': info_LONG,
                                          'event_DEPTH': info_DEPTH,
                                          'STA': info_STA,
                                          'CHA': info_CHA_E,
                                          'station_DATE': info_DATE,
                                          'station_TIME': info_TIME,
                                          'MAG': info_MAG,
                                          'MAG_D': info_MAG_D,
                                          'LABEL_E': info_LABEL,
                                          'LABEL_W': 'none',
                                          'LABEL_D': 'none'
                                          }

                            if ext_file_E != [] and ext_file_N != [] and ext_file_Z != []:
                                st_E = read(ext_file_E[0])  ## file reading
                                st_N = read(ext_file_N[0])  ## file reading
                                st_Z = read(ext_file_Z[0])  ## file reading

                                tr_E, sta_LAT, sta_LONG, sta_DEPTH, sample_rate = extract_seed(st_E, time_DATE_E,
                                                                                               save_dir, info_STA,
                                                                                               info_CHA_E, info_DATE_E,
                                                                                               info_TIME_T,
                                                                                               extract_time,
                                                                                               time_DATE_E, event_info,
                                                                                               'E')
                                tr_N, _, _, _, _ = extract_seed(st_N, time_DATE_E, save_dir, info_STA, info_CHA_N,
                                                                info_DATE_E, info_TIME_T,
                                                                extract_time, time_DATE_E, event_info, 'N')
                                tr_Z, _, _, _, _ = extract_seed(st_Z, time_DATE_E, save_dir, info_STA, info_CHA_Z,
                                                                info_DATE_E, info_TIME_T,
                                                                extract_time, time_DATE_E, event_info, 'Z')

                                if sta_LAT == 0.0 or sta_LONG == 0.0:
                                    continue
                                if sample_rate <= 20:
                                    continue

                                if len(tr_E) == 7000 and len(tr_N) == 7000 and len(tr_Z) == 7000:
                                    event_info['tr_E'] = tr_E
                                    event_info['tr_N'] = tr_N
                                    event_info['tr_Z'] = tr_Z
                                    event_info['sta_LAT'] = sta_LAT
                                    event_info['sta_LONG'] = sta_LONG
                                    event_info['sta_DEPTH'] = sta_DEPTH

                                    name_seed = event_info['STA'] + '_' + event_info['CHA']  ### mseed filename
                                    event_save_dir = save_dir + '/event/' + '{0:03}'.format(idx) + '/'
                                    if not os.path.isdir(event_save_dir):
                                        os.makedirs(event_save_dir)
                                    gen_name1 = event_save_dir + name_seed + '.npz'

                                    np.savez(gen_name1, **event_info)
        else:
            temp = f.readline()

    f.close()

    return


current_path = u"/sde1/2018_list"
each_dir = [u"1_국내지진", u"3_미소지진", u"4_인공지진"]

upper_dir = '/sdd1/Eq2020_multisite_0925/2018Data2/'
odata_dir = "/media/super/4af6ecf9-b4dc-4ffd-9da0-0f1667a69e01/2018/"

dir1_name = upper_dir + '/0'
dir2_name = upper_dir + '/1'
dir3_name = upper_dir + '/2'
dir4_name = upper_dir + '/3'

if not (os.path.isdir(dir1_name)):
    os.makedirs(os.path.join(dir1_name))
if not (os.path.isdir(dir2_name)):
    os.makedirs(os.path.join(dir2_name))
if not (os.path.isdir(dir3_name)):
    os.makedirs(os.path.join(dir3_name))
if not (os.path.isdir(dir4_name)):
    os.makedirs(os.path.join(dir4_name))

UTC_REF = '2018-01-01T00:00:01'
time = UTCDateTime(UTC_REF)
Ref_julday = time.julday

event_index = -1
for dir_i in each_dir:
    print(os.path.join(current_path, dir_i))
    data_path = os.path.join(current_path, dir_i)
    event_index = event_index + 1

    idx = 0
    for (path, dir, files) in os.walk(data_path):
        for filename in files:
            ext = os.path.splitext(filename)[0]
            if ext.find('arrival') != (-1):
                print("%s/%s" % (path, filename))
                mafe_EI(os.path.join(path, filename), event_index, upper_dir, odata_dir, idx)
                idx += 1


