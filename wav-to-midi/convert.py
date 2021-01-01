# !/usr/bin/env python3
import os
import subprocess
import shutil

# pip install audio-to-midi

abs_path = os.path.abspath('..')  # string path


def wav_convert():
    """
    convert all .wav files in cwd to .mid & move .wav to folder
    """
    # create dir if dne
    if not os.path.isdir(abs_path + '/wav-files'):
        subprocess.run(["mkdir", "wav-files"])
        os.system("echo directory <wav-files> created")

    # find and convert.wav
    for file in os.listdir('..'):
        if os.path.isdir(file):
            pass;
        elif not file.endswith('.wav'):
            print("file <", file, "> must be .wav")
        else:
            print("processing <" + file + ">")
            path = "\"" + file + "\""
            subprocess.run(["audio-to-midi ./" + path], shell=True)
            shutil.move(file, './wav-files')


### main
os.system("echo .....converting files.....")
wav_convert()
