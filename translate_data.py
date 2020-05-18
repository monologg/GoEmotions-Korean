import os
import time
import argparse
from tqdm import tqdm
from googletrans import Translator as GoogleTranslator

ORIG_DATA_DIR = os.path.join("goemotions", "data")
DATA_DIR = "data"

TRAIN_FILE = "train.tsv"
DEV_FILE = "dev.tsv"
TEST_FILE = "test.tsv"

TEXT_MAX_LENGTH = 5000  # google translate allows maximum size of 5000 for one request
GOOGLE_TIME_TO_SLEEP = 1.5


def make_chunks(sentence_lst):
    """
    Chunk is a sentence that is not longer than TEXT_MAX_LENGTH
    By looping the list of sentences, we will make a new chunk which is not longer than TEXT_MAX_LENGTH, while as long as possible
    """
    input_chunk_lst = []
    chunk = ""
    for sentence in sentence_lst:
        sentence = sentence.strip()
        # https://www.reddit.com/r/OutOfTheLoop/comments/9abjhm/what_does_x200b_mean/
        sentence = sentence.replace("&#x200B;", "")  # This one makes error
        sentence = sentence + "\r\n"
        if len((chunk.rstrip() + sentence).encode('utf-8')) > TEXT_MAX_LENGTH:
            input_chunk_lst.append(chunk.rstrip())
            chunk = sentence
        else:
            chunk = chunk + sentence
    input_chunk_lst.append(chunk.rstrip())
    return input_chunk_lst


def get_sentence_lst(file_path):
    sentence_lst = []
    label_lst = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            items = line.split("\t")
            sentence = items[0].strip()
            label = items[1]
            sentence_lst.append(sentence)
            label_lst.append(label)
    return sentence_lst, label_lst


def google_translate(sentence_lst):
    input_chunk_lst = make_chunks(sentence_lst)
    trans = GoogleTranslator()
    translated_sentence_lst = []

    for en_chunk in tqdm(input_chunk_lst):
        kr_chunk = trans.translate(en_chunk, src='en', dest='ko')
        kr_chunk = kr_chunk.text
        kr_sentences = kr_chunk.split("\r\n")
        if kr_sentences[-1] == "":
            kr_sentences = kr_sentences[:-1]
        time.sleep(GOOGLE_TIME_TO_SLEEP)

        translated_sentence_lst.extend(kr_sentences)

    return translated_sentence_lst


def make_translate_data(orig_file_path, translated_file_path):
    sentence_lst, label_lst = get_sentence_lst(orig_file_path)
    translate_sentence_lst = google_translate(sentence_lst)

    assert len(translate_sentence_lst) == len(label_lst)

    with open(translated_file_path, "w", encoding="utf-8") as f:
        for (translated_sent, label) in zip(translate_sentence_lst, label_lst):
            f.write("{}\t{}\n".format(translated_sent, label))

    print("Translating {} done".format(orig_file_path))


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    make_translate_data(
        os.path.join(ORIG_DATA_DIR, TRAIN_FILE),
        os.path.join(DATA_DIR, TRAIN_FILE)
    )

    make_translate_data(
        os.path.join(ORIG_DATA_DIR, DEV_FILE),
        os.path.join(DATA_DIR, DEV_FILE)
    )

    make_translate_data(
        os.path.join(ORIG_DATA_DIR, TEST_FILE),
        os.path.join(DATA_DIR, TEST_FILE)
    )
