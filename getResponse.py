import numpy as np
import pandas as pd
import nltk
import random
import json
import re
import string
import itertools

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# start preprocessing text
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopWordFactory = StopWordRemoverFactory()

# membuat text menjadi huruf kecil semua
def lower(text):
    return text.lower()

# replace text yg bukan termasuk ascii
def non_ascii(text):
    return text.encode('ascii', 'replace').decode('ascii')

# hapus tanda baca
def remove_punctuation(text):
    remove = string.punctuation
    remove = remove.replace("_", "")
    pattern = r"[{}]".format(remove)
    return re.sub(pattern, "", text)

# hapus spasi diawal dan akhir
def remove_whitespace_LT(text):
    return text.strip()

# hapus spasi double
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

# hapus tab
def remove_tab(text):
    return text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")

# hapus multiple tab
def remove_tab2(text):
    return re.sub('\s+',' ',text)

# hapus angka
def remove_angka(text):
    return re.sub(r"\d+", "", text)

# hapus emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# hapus titik banyak (...)
def remove_excessive_dot(text):
    return text.replace('..'," ")

# stopwords
def stopwords(text):
    more_stopword = ['sih', 'nya','rt','loh','lah', 'dd', 'mah', 'nye', 'eh', 'ehh', 'ah', 'yang','yg']
    data = stopWordFactory.get_stop_words()
    data.remove('tidak')
    stopwords_sastrawi = stopWordFactory.create_stop_word_remover()

    dictionary = ArrayDictionary(data+more_stopword)
    str_stopwords = StopWordRemover(dictionary)
    return str_stopwords.remove(text)

# stemming
def stemming(text):
    return stemmer.stem(text)

indo_slang_word = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")

# convert df slang to dict
df_slang = indo_slang_word[['slang','formal']]
slang_dict = dict(df_slang.values)

# function to replace slang word
def replace_slang_word(string):
    new_string = " ".join(slang_dict.get(word, word) for word in string.split(' '))
    return new_string

# end preprocessing text

# start get dataset
with open('datasets/intent_datasets.json') as file:
    intents = json.load(file)

df = pd.DataFrame(intents['intents'])
labels = df['tag'].unique().tolist()
labels = [s.strip() for s in labels]
label2id = {label:id for id, label in enumerate(labels)}
# end get dataset

# start predict response
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import pipeline
model_path = "chatbot"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def preprocessing_user_input(text):
    text = lower(text)
    text = non_ascii(text)
    text = remove_emoji(text)
    text = remove_tab(text)
    text = remove_tab2(text)
    text = remove_punctuation(text)
    text = remove_excessive_dot(text)
    text = remove_angka(text)
    text = remove_whitespace_LT(text)
    text = remove_whitespace_multiple(text)
    text = replace_slang_word(text)
    text = stopwords(text)
    text = stemming(text)
    return text

def chat(text):
    # text = input("User: ").strip()
    text = preprocessing_user_input(text)

    score = chatbot(text)[0]['score']

    if score < 0.2:
        response = "Maaf, saya tidak bisa menjawabnya"
        return response

    label = label2id[chatbot(text)[0]['label']]
    response = random.choice(intents['intents'][label]['responses'])
    return response