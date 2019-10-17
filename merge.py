import shelve
import os
from shelve_reader import ShelveReader
import pandas as pd

def merge_file(mfcc_file, points_file):
    with shelve.open('extracted_data/' + mfcc_file, 'r') as mfcc, shelve.open('extracted_data/' + points_file, 'r') as points:
        res = []
        audio_reader = ShelveReader(mfcc, 'audio')
        video_reader = ShelveReader(points, 'video')
        for a_time, a in audio_reader:
            for v_time, v in video_reader:
                if round(a_time - v_time) == 0:
                    res.append([a_time, [a, v]])
                    break
                elif round(a_time - v_time) < 0:
                    video_reader.decrease_iter()
                    break

        #for i, (time, (a, v)) in enumerate(res):
        #    print(i)
        #    a_test = []
        #    v_test = []
        #    for i in range(len(mfcc)):
        #        if mfcc[str(i)][0] == time:
        #            a_test.append(mfcc[str(i)][1])
        #            
        #    for i in range(len(points)):
        #        if points[str(i)][0] == time:
        #            v_test.append(points[str(i)][1])
        #            
        #    assert a in a_test and v in v_test, 'Invalid merge'

    return res

def merge_to_shelve():
    list_audio = []
    list_video = []
    for f_name in sorted(os.listdir('extracted_data')):
        if f_name.startswith('audio'):
            list_audio.append(f_name)
        elif f_name.startswith('video'):
            list_video.append(f_name)

    res = []

    for num, (audio, video) in enumerate(zip(list_audio, list_video)):
        with shelve.open('extracted_data/merge_{}.txt'.format(num), 'c') as db:
            for i, x in enumerate(merge_file(audio, video), len(db)):
                res.append(x)
                db[str(i)] = x

def merge_to_csv():
    list_audio = []
    list_video = []
    for f_name in sorted(os.listdir('extracted_data')):
        if f_name.startswith('audio'):
            list_audio.append(f_name)
        elif f_name.startswith('video'):
            list_video.append(f_name)

    columns = ['time'] + ['mfcc_{}'.format(i) for i in range(13)] + ['point_{}'.format(i) for i in range(136)]
    data = {col_name: [] for col_name in columns}

    for num, (audio, video) in enumerate(zip(list_audio, list_video)):
        for time, (a, v) in merge_file(audio, video):
            data['time'].append(time)
            for i, coef in enumerate(a):
                data['mfcc_{}'.format(i)].append(coef)
            for i, point in enumerate(v):
                data['point_{}'.format(2 * i)].append(point[0])
                data['point_{}'.format(2 * i + 1)].append(point[1])

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv('extracted_data/merge_{}.csv'.format(num), index=False)

if __name__ == '__main__':
    merge_to_csv()