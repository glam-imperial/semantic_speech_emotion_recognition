import csv
import os
import sys
import time
sys.path.append(".")

from aeneas.audiofile import AudioFile
from aeneas.exacttiming import TimeInterval, TimeValue
from aeneas.executetask import ExecuteTask
from aeneas.task import Task

from helper import get_pathlist_from_dir


AVEC_DIR = '/path/to/AVEC2017_SEWA/' 

AUDIO_DIR = AVEC_DIR + 'audio/'
TRANSCRIPT_DIR = AVEC_DIR + 'transcriptions/'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENT_DIR = OUT_DIR + '/segmentation/'


def execute_task(directory, filename):
    # Create Task object
    config_string = u"task_language=deu|is_text_type=mplain|os_task_file_format=csv|os_task_file_levels=3"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = directory + filename + '.wav'
    task.text_file_path_absolute = directory + filename + '.txt'
    task.sync_map_file_path_absolute = "%s/%s.csv" % (SEGMENT_DIR, filename)
    
    # Process Task
    ExecuteTask(task).execute()
    
    # output sync map to file
    task.output_sync_map_file()


def trim_audio(audio_file, sent):
    audio = AudioFile(file_path=AUDIO_DIR + audio_file + '.wav')
    audio.read_properties()
    audio.read_samples_from_file()
    
    # Extract sentence information
    start, end, transcript = sent
    
    start = TimeValue(start)
    end = TimeValue(end)
    time_interval = TimeInterval(start, end)
    
    # Trim audio by sentence
    audio.trim(begin=start, length=time_interval.length)
    assert audio.audio_length - time_interval.length < 0.001
    
    fo = audio_file + '_' + str(int(time.time()))
    audio.write(SEGMENT_DIR + fo + '.wav')
    with open(SEGMENT_DIR + fo + '.txt', 'w') as tmp_transcript:
        tmp_transcript.write(transcript)
    
    return fo


def load_transcript(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        
        sentences = []
        for row in reader:
            transcript = row[2].strip('\'')  # Strip '' at beginning & end
            
            punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'  # Exclude <>
            transcript_clean = transcript.translate(str.maketrans('', '', punctuation))
            words = transcript_clean.lower().split()
            
            if transcript != '<filler>' and transcript_clean != '':
                sentences.append([row[0], row[1], transcript_clean])
    
    return sentences


def clean_word(word):
    word = word.lower()
    
    punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'  # Exclude <>
    word = word.translate(str.maketrans('', '', punctuation))
    
    return word


def delete_segmentation(tmp):
    for ext in ['.wav', '.txt', '.csv']:
        os.remove(SEGMENT_DIR + tmp + ext)


def mapping_generator():
    pathlist = get_pathlist_from_dir(TRANSCRIPT_DIR)
    
    for path in pathlist:
        p = path.rsplit('/', 1)
        filename = str(p[-1])[:-4]
        print('Create synchronisation map for %s' % filename)
        
        fo = open(SEGMENT_DIR + filename + '.csv', 'w')
        
        sentences = load_transcript(path)
        for sid, sent in enumerate(sentences):
            time_to_shift = float(sent[0])
            
            tmp_file = trim_audio(filename, sent)
            
            # Mapping audio with transcript
            execute_task(SEGMENT_DIR, filename=tmp_file)
            
            # Write to final synchronisation mapping
            with open(SEGMENT_DIR + tmp_file + '.csv') as tmp:
                for row in csv.reader(tmp):
                    id, start, end, word = row
                    
                    id = 's' + str(sid) + 'w' + id.split('w')[1]
                    
                    start = round((float(start) + time_to_shift), 4)
                    end = round((float(end) + time_to_shift), 4)
                    
                    word = clean_word(word)
                    
                    new_row = ";".join([id, str(start), str(end), word])
                    fo.write(new_row + '\n')
            
            # Delete tmp files
            delete_segmentation(tmp_file)
        
        fo.close()


if __name__ == '__main__':
    mapping_generator()