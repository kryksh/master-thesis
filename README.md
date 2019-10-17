# master-thesis
Исходные файлы представлены в виде audio_xxx.wav, video_xxx.mp4, где xxx - номер файла (например audio_7.wav, video_7.mp4).
pre_audio.py и pre_video.py преобразуют файлы audio_xxx.wav, video_xxx.mp4 в файлы audio_xxx.txt, video_xxx.txt, которые помщаются  в каталог extracted_data.
audio_xxx.wav хранит наборы MFCC-коэффициентов, video_xxx.mp4 хранит наборы координат ключевых точек лицевых ориентиров. 
merge.py объединяет файлы audio_xxx.txt, video_xxx.txt в файл merge.txt
В merge.txt хранится информация о сопоставленных по времени наборах MFCC-коэффициентов и точек.
preprocessing.py нормализация и разбиение выборки на трейн и тест.
NN_COLAB_TPU.ipynb построение модели и отрисовка полученных результатов.
