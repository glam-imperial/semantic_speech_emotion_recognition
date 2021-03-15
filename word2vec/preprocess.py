import csv
import nltk
import os
import re
import string
import time

from nltk.tokenize import RegexpTokenizer

# Define Paths
data_dir = '/path/to/AVEC2017_SEWA' + '/transcriptions'
swc_data_dir = '/path/to/'+'swc/german/'

# word2vec corpus directory
text_corpus_dir = '/corpus/raw/'
text_corpus_processed_dir = '/corpus/processed/'

stop_words = ["a","ab","aber","ach","acht","achte","achten","achter","achtes","ag","alle","allein","allem","allen","aller","allerdings","alles","allgemeinen","als","also","am","an","ander","andere","anderem","anderen","anderer","anderes","anderm","andern","anderr","anders","au","auch","auf","aus","ausser","ausserdem","außer","außerdem","b","bald","bei","beide","beiden","beim","beispiel","bekannt","bereits","besonders","besser","besten","bin","bis","bisher","bist","c","d","d.h","da","dabei","dadurch","dafür","dagegen","daher","dahin","dahinter","damals","damit","danach","daneben","dank","dann","daran","darauf","daraus","darf","darfst","darin","darum","darunter","darüber","das","dasein","daselbst","dass","dasselbe","davon","davor","dazu","dazwischen","daß","dein","deine","deinem","deinen","deiner","deines","dem","dementsprechend","demgegenüber","demgemäss","demgemäß","demselben","demzufolge","den","denen","denn","denselben","der","deren","derer","derjenige","derjenigen","dermassen","dermaßen","derselbe","derselben","des","deshalb","desselben","dessen","deswegen","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","dir","doch","dort","drei","drin","dritte","dritten","dritter","drittes","du","durch","durchaus","durfte","durften","dürfen","dürft","e","eben","ebenso","ehrlich","ei","ei,","eigen","eigene","eigenen","eigener","eigenes","ein","einander","eine","einem","einen","einer","eines","einig","einige","einigem","einigen","einiger","einiges","einmal","eins","elf","en","ende","endlich","entweder","er","ernst","erst","erste","ersten","erster","erstes","es","etwa","etwas","euch","euer","eure","eurem","euren","eurer","eures","f","folgende","früher","fünf","fünfte","fünften","fünfter","fünftes","für","g","gab","ganz","ganze","ganzen","ganzer","ganzes","gar","gedurft","gegen","gegenüber","gehabt","gehen","geht","gekannt","gekonnt","gemacht","gemocht","gemusst","genug","gerade","gern","gesagt","geschweige","gewesen","gewollt","geworden","gibt","ging","gleich","gott","gross","grosse","grossen","grosser","grosses","groß","große","großen","großer","großes","gut","gute","guter","gutes","h","hab","habe","haben","habt","hast","hat","hatte","hatten","hattest","hattet","heisst","her","heute","hier","hin","hinter","hoch","hätte","hätten","i","ich","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","im","immer","in","indem","infolgedessen","ins","irgend","ist","j","ja","jahr","jahre","jahren","je","jede","jedem","jeden","jeder","jedermann","jedermanns","jedes","jedoch","jemand","jemandem","jemanden","jene","jenem","jenen","jener","jenes","jetzt","k","kam","kann","kannst","kaum","kein","keine","keinem","keinen","keiner","keines","kleine","kleinen","kleiner","kleines","kommen","kommt","konnte","konnten","kurz","können","könnt","könnte","l","lang","lange","leicht","leide","lieber","los","m","machen","macht","machte","mag","magst","mahn","mal","man","manche","manchem","manchen","mancher","manches","mann","mehr","mein","meine","meinem","meinen","meiner","meines","mensch","menschen","mich","mir","mit","mittel","mochte","mochten","morgen","muss","musst","musste","mussten","muß","mußt","möchte","mögen","möglich","mögt","müssen","müsst","müßt","n","na","nach","nachdem","nahm","natürlich","neben","nein","neue","neuen","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nie","niemand","niemandem","niemanden","noch","nun","nur","o","ob","oben","oder","offen","oft","ohne","ordnung","p","q","r","recht","rechte","rechten","rechter","rechtes","richtig","rund","s","sa","sache","sagt","sagte","sah","satt","schlecht","schluss","schon","sechs","sechste","sechsten","sechster","sechstes","sehr","sei","seid","seien","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","selbst","sich","sie","sieben","siebente","siebenten","siebenter","siebentes","sind","so","solang","solche","solchem","solchen","solcher","solches","soll","sollen","sollst","sollt","sollte","sollten","sondern","sonst","soweit","sowie","später","startseite","statt","steht","suche","t","tag","tage","tagen","tat","teil","tel","tritt","trotzdem","tun","u","uhr","um","und","und?","uns","unse","unsem","unsen","unser","unsere","unserer","unses","unter","v","vergangenen","viel","viele","vielem","vielen","vielleicht","vier","vierte","vierten","vierter","viertes","vom","von","vor","w","wahr?","wann","war","waren","warst","wart","warum","was","weg","wegen","weil","weit","weiter","weitere","weiteren","weiteres","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wer","werde","werden","werdet","weshalb","wessen","wie","wieder","wieso","will","willst","wir","wird","wirklich","wirst","wissen","wo","woher","wohin","wohl","wollen","wollt","wollte","wollten","worden","wurde","wurden","während","währenddem","währenddessen","wäre","würde","würden","x","y","z","z.b","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zu","zuerst","zugleich","zum","zunächst","zur","zurück","zusammen","zwanzig","zwar","zwei","zweite","zweiten","zweiter","zweites","zwischen","zwölf","über","überhaupt","übrigens"]


###########################
# Process AVEC csv files
###########################
def process_avec(dataset=None):
    """
    Read transcription files (.csv) and extract words from them in order
    :param dataset: train/evaluation/test or all dataset
    :return: a list of words
    """
    train_pathlist, val_pathlist, test_pathlist = _get_pathlist()

    pathlist = []
    if dataset is None:
        pathlist.extend(train_pathlist + val_pathlist + test_pathlist)
    elif dataset == 'train':
        pathlist = train_pathlist
    elif dataset == 'val':
        pathlist = val_pathlist
    elif dataset == 'test':
        pathlist = test_pathlist

    text_words = _process_csv_files(pathlist)
    
    return text_words


def _get_pathlist():
    """
    Get all pathlists of csv files
    :return: 3 lists of filepaths (train, evaluation, test)
    """
    train_pathlist = []
    val_pathlist = []
    test_pathlist = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if filename.startswith('Train'):
            train_pathlist.append(path)
        elif filename.startswith('Devel'):
            val_pathlist.append(path)
        elif filename.startswith('Test'):
            test_pathlist.append(path)

    return train_pathlist, val_pathlist, test_pathlist


def _process_csv_files(pathlist):
    """
    Extract words from all csv files
    :param pathlist: a list of paths to csv files
    :return: a list of words
    """
    text_words = []
    for filepath in pathlist:
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for row in reader:
                transcript = ', '.join(row[2:])

                punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'  # Exclude <>
                transcript_clean = transcript.translate(str.maketrans('', '', punctuation))

                words = transcript_clean.lower().split()
                text_words.extend(words)

    return text_words


#####################################
# Process txt files from SWC corpus (1)
#####################################
def process_swc():
    """
    Read wiki text files from SWC corpus and extract words in order
    :return: list of words
    """
    pathlist = _get_pathlist_swc()
    text_words = _process_txt_files(pathlist)

    return text_words


def _get_pathlist_swc():
    pathlist = []
    for subdir, dirs, files in os.walk(swc_data_dir):
        for file in files:
            if file == 'wiki.txt':
                pathlist.append(os.path.join(subdir, file))

    return pathlist


def _process_txt_files(pathlist):
    """
    Process text files from SWC corpus
    :param pathlist: list of file paths
    :return: a list of words extracted from all files in pathlist
    """
    text_words = []

    for path in pathlist:
        file = open(path, 'r')
        raw_data = file.read()
        file.close()

        data_clean = raw_data.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = data_clean.lower().split()  # Split document into words

        # Remove stop words
        # stop_words = nltk.corpus.stopwords.words('german')

        # stop_words = get_stop_words('german')
        # stop_words.extend(stop_words_2)

        words = [w for w in words if w not in stop_words]  # Remove stop words
        words_clean = [w for w in words if len(w)>1 and w.isalpha()]  # Remove all words that are not letters
        text_words.extend(words_clean)

    return text_words


#####################################
# Process txt files from SWC corpus (2)
#####################################
def preprocess_wiki_files():
    dir = os.path.dirname(os.path.realpath(__file__))
    raw_dir = dir + text_corpus_processed_dir

    articles = []
    for filename in os.listdir(raw_dir):
        filepath = raw_dir + filename
        sents_tokenized, _ = _split_sentences_from_filepath(filepath)
        articles.append(sents_tokenized)

    return articles


def _split_sentences_from_filepath(filepath):
    sents_untokenized = []
    sents_tokenized = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) <= 10:
                continue

            if line.startswith('[[') or line.startswith('=='):
                continue

            sentences = nltk.tokenize.sent_tokenize(line, language='german')
            for s in sentences:
                s = _remove_brackets(s)

                tokenizer = RegexpTokenizer('\w+')
                words = tokenizer.tokenize(s)
                words = [word.lower() for word in words if word.isalpha() and len(word) > 1]
                if len(words) > 5:  # A sentence has at least 5 words
                    sents_untokenized.append(s)
                    sents_tokenized.append(words)

    return sents_tokenized, sents_untokenized


def _remove_brackets(sent):
    """
    :param sent: a sentence
    :return: a sentence with brackets ( [[..]], [..], {{..}}, {..}, (..) ) removed
    """
    sent = re.sub(r'\[\[([^\]]*)\]\]', r'\1', sent)
    sent = re.sub(r'\[([^\]]*)\]', r'\1', sent)
    sent = re.sub(r'\{\{([^\]]*)\}\}', r'\1', sent)
    sent = re.sub(r'\{([^\]]*)\}', r'\1', sent)
    sent = re.sub(r'\(([^\]]*)\)', r'\1', sent)

    return sent


def _extract_text_from_swc():
    pathlist = _get_pathlist_swc()
    dir = os.path.dirname(os.path.realpath(__file__))

    for path in pathlist:
        filename = path.split('/')[-2]

        f1 = open(path, 'r')
        content = f1.read()
        f1.close()

        new_path = dir + text_corpus_dir + str(filename) + '.txt'
        f2 = open(new_path, 'w')
        f2.write(content)
        f2.close()


def _write_preprocessed_sents_to_file():
    dir = os.path.dirname(os.path.realpath(__file__))
    raw_dir = dir + text_corpus_dir
    processed_dir = dir + text_corpus_processed_dir

    print('Start processing text files ...')
    tic = time.time()

    for filename in os.listdir(raw_dir):
        filepath = raw_dir + filename
        _, sents_untokenized = _split_sentences_from_filepath(filepath)

        f = open(processed_dir + filename, 'w')
        for sent in sents_untokenized:
            f.write(sent + '\n')
        f.close()

    print('Done processing in {%.4f} sec! Files are saved in path %s' % (time.time() - tic, processed_dir))


def main():
    _extract_text_from_swc()
    _write_preprocessed_sents_to_file()
