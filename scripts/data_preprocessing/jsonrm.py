import sys
import os
import struct
import json

from glob import glob
from pathlib import Path
from tqdm import tqdm


ordered_blendshapes = ['eyeBlink_L', 'eyeLookDown_L', 'eyeLookIn_L', 'eyeLookOut_L', 'eyeLookUp_L', 'eyeSquint_L', 'eyeWide_L', 'eyeBlink_R', 'eyeLookDown_R', 'eyeLookIn_R', 'eyeLookOut_R', 'eyeLookUp_R', 'eyeSquint_R', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight', 'mouthSmile_L', 'mouthSmile_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthDimple_L', 'mouthDimple_R', 'mouthStretch_L', 'mouthStretch_R', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthPress_L', 'mouthPress_R', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'browDown_L', 'browDown_R', 'browInnerUp', 'browOuterUp_L', 'browOuterUp_R', 'cheekPuff', 'cheekSquint_L', 'cheekSquint_R', 'noseSneer_L', 'noseSneer_R', 'tongueOut']


def write_value(file, type, value):
    format = '<{}'.format(type)
    file.write(struct.pack(format, value))

def write_list(file, type, values):
    write_value(file, 'i', len(values))
    format = '<{}{}'.format(len(values), type)
    file.write(struct.pack(format, *values))

def write_defaults(file, json_data):
    write_value(file, 'i', 23592)
    write_list(file, 'h', json_data['indices'])
    write_list(file, 'f', json_data['uv'])

def write_frame(file, json_data):
    write_value(file, 'i', 14980)
    write_list(file, 'f', [json_data['blendshapes'][name] for name in ordered_blendshapes])
    write_list(file, 'f', json_data['transform'])
    write_list(file, 'f', json_data['intrinsics'])
    write_list(file, 'f', json_data['vertices'])
    write_value(file, 'd', json_data['timestamp'])
    write_value(file, 'd', json_data['global_time'])


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = ''

    json_files = glob(os.path.join(path, '**/*.jpg'), recursive=True)
    mp4_files = glob(os.path.join(path, '**/*.csv'), recursive=True)
    #gar_files = glob(os.path.join(path, '**/._*'), recursive=True)

    json_files_per_sequences = {}
    for json_file in json_files:
        json_file_path = Path(json_file)
        if json_file_path.stem == 'start.json' or json_file_path.stem == 'end.json' or '__processed__' in json_file:
            continue
        dirname = json_file_path.parent
        if dirname not in json_files_per_sequences:
            json_files_per_sequences[dirname] = []
        json_files_per_sequences[dirname].append(json_file)
        
    total_frames = 0
    total_sequences = 0

    for dirname, json_files_per_sequence in tqdm(json_files_per_sequences.items()):
        defaults_path = None
        frame_paths = []

        for json_file in json_files_per_sequence:
            os.remove(json_file)

        if defaults_path is None and len(frame_paths) == 0:
            print('json files not found for: {}'.format(dirname))
            continue


    mp4_files_per_sequences = {}
    for mp4_file in mp4_files:
        mp4_file_path = Path(mp4_file)
        if mp4_file_path.stem == 'start.json' or mp4_file_path.stem == 'end.json' or '__processed__' in mp4_file:
            continue
        dirname = mp4_file_path.parent
        if dirname not in mp4_files_per_sequences:
            mp4_files_per_sequences[dirname] = []
        mp4_files_per_sequences[dirname].append(mp4_file)
        
    total_frames = 0
    total_sequences = 0

    for dirname, mp4_files_per_sequence in tqdm(mp4_files_per_sequences.items()):
        defaults_path = None
        frame_paths = []

        for mp4_file in mp4_files_per_sequence:
            os.remove(mp4_file)

        if defaults_path is None and len(frame_paths) == 0:
            print('mp4 files not found for: {}'.format(dirname))
            continue



