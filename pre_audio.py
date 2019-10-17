import librosa
import shelve
import os

d = 1/30
def get_mfcc(audio_file, duration=d):
    print('start extract mfcc') 
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(sr*duration), n_fft=int(sr*duration)*4, n_mfcc=13) # 4 потому что дефолтные значения hop_length = 512 и n_fft = 2048, а 2048 / 512 = 4, кек
    print('end extract mfcc')

    list_mfcc = [[duration*1000*i, x.tolist()] for i, x in enumerate(mfcc.T, 1)]

    with shelve.open('extracted_data/' + audio_file.split('/')[-1].split('.')[0] + '.txt', 'c') as db:
        for i, x in enumerate(list_mfcc):
            db[str(i)] = x

    return list_mfcc

def extract_mfcc():
    res = []
    for i in range(sum(f_name.startswith('audio') for f_name in os.listdir('data'))):
        get_mfcc('data/audio_{}.wav'.format(i))

if __name__ == '__main__':
    extract_mfcc()