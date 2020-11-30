import os
import json
import pickle
import sys
import signal
import traceback
import re
import pandas as pd
import inspect
import time

import torch
from tabulate import tabulate

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, grandparentdir)

from fast_bert.prediction import BertNERPredictor
from fast_bert.utils.spellcheck import BingSpellCheck
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

prefix = "./opt/ml/"

MODEL_PATH = os.path.join(prefix, "model")
DATA_PATH = os.path.join(prefix, "input/data")

class Scorer(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_predictor_model(cls):

        if cls.model is None:
            with open(os.path.join(MODEL_PATH, "model_config.json")) as f:
                model_config = json.load(f)

            predictor = BertNERPredictor(
                os.path.join(MODEL_PATH, "model_out"),
                label_path=MODEL_PATH,
                model_type=model_config["model_type"],
                do_lower_case=model_config.get("do_lower_case", "True") == "True",
                use_fast_tokenizer=model_config.get("use_fast_tokenizer", "True") == "True",
                device='cpu'
            )
            cls.model = predictor
        return cls.model

    @classmethod
    def predict(cls, text):

        predictor_model = cls.get_predictor_model()
        prediction = predictor_model.predict(text)

        return prediction

model = Scorer.get_predictor_model()
data = pd.read_csv(os.path.join(DATA_PATH,'test.csv'))
text = data["text"].loc[0:1].tolist() 
text = ["Austria's Chancellor, Sebastian said that the victims included \
\"an elderly man, an elderly woman, a young passer-by and a waitress.\" In  \
addition to the four civilians, a gunman was shot dead by police. \
Authorities have identified the attacker as Fejzulai Kujtim, a 20-year-old \
Austrian man from 33 miles west of Vienna. Kujtim was a supporter of Islamic State, \
who was sentenced to 22 months in prison on April 25, 2019 for attempting to travel \
to Syria to join ISIS, Minister Karl Nehammer told state news agency APA. On December 5, \
he was released early on parole, it reports. Police in Vienna have arrested 14 people \
and searched 18 properties in relation to the attack. Initial reports on Monday night \
said multiple gunmen opened fire at six locations in the city center, as residents savored \
 the final hours of freedom before the imposition of a nationwide lockdown. \
But authorities have since cast doubt on whether the man police shot was part \
of a larger group. Austrian police said Tuesday morning \"they assume that there  \
were more attackers\" said at the press conference, \"it can't be excluded that \
there were more attackers.\" Austrian authorities told CNN they cannot rule \
if a second suspect is on the run. Vienna police spokesperson Christopher \
Verhnjak said police had been informed by witnesses there could be more than one \
attacker. Police are investigating and advising people to stay home until they are \
sure there isn't a suspect in hiding. Armed forces have been deployed in Vienna to \
help secure the situation, with authorities indicating earlier in the evening that \
at least one gunman remains on the loose. Residents of Vienna have been asked to stay",
"at home or in a safe place and follow the news. Authorities have abandoned compulsory \
school attendance and asked citizens to avoid the city center for fears of another attacker \
still at large. Earlier, Vienna police said that SWAT teams entered the gunman's apartment \
using explosives and a search of its surroundings was underway. Police have also received \
more than 20,000 videos from members of the public following the attack. The initial \
attack, which began around 8 p.m., was centered on the busy shopping and dining \
district near Vienna's main synagogue, Seitenstettengasse Temple, which was closed. \
The five other locations were identified as Salzgries, Fleischmarkt, Bauernmarkt, \
Graben and, Morzinplatz near the Temple, according to an Austrian law enforcement \
source speaking to journalists on Tuesday. Vienna mayor Michael Ludwig said shots \
appeared to be fired at random, as people dined and drank outside due to the warm \
weather and virus concerns. Julia Hiermann, who lives in Vienna, was having drinks \
with a friend when the shooting began. ","I love Elizabeth" ]

t = []
for i in range(10):
    t.extend(text)
s = time.time()
ners = model.predict_batch(t)   
print(f'total time took {(time.time()-s):.3f} seconds')

# start = time.time()
# ners = model.predict_batch(text)
# end = time.time()
# print(f'the data size on disk is {sys.getsizeof(text)/1000} KB')
# print(f'it took {(end-start):.0f} seconds')

# ner_map = []
# for n in ners:
#     for r in n["results"]:
#         ner_map.append([r['word'],r['entity'],r['score']])
# print(tabulate(ner_map))

# text = ["I went to 中国 with my friends 鲍勃和亚历克斯"]
# ners = model.predict_batch(text)
# ner_map = []
# for n in ners:
#     for r in n["results"]:
#         ner_map.append([r['word'],r['entity'],r['score']])
# print(tabulate(ner_map))
