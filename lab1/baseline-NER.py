#! /usr/bin/python3

from evaluator import load_gold_NER, load_gold_NER_ext
import sys
from os import listdir, system
import re
from math import sqrt


from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import evaluator

# dictionary containig information from external knowledge resources
# WARNING: You may need to adjust the path to the resource files
external = {}
with open("resources/HSDB.txt") as h:
    for x in h.readlines():
        external[x.strip().lower()] = "drug"
with open("resources/DrugBank.txt") as h:
    for x in h.readlines():
        (n, t) = x.strip().lower().split("|")
        external[n] = t


# --------- tokenize sentence -----------
# -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    for t in word_tokenize(txt):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
    return tks

# -----------------------------------------------
# -- check if a token is a drug part, and of which type


suffix_drug = ['azole', 'amine', 'farin', 'idine', 'mycin',
               'ytoin', 'goxin', 'navir', 'etine', 'lline']
suffix_brand = ['pirin', 'DOCIN', 'TAXOL', 'GASYS', 'VIOXX',
                'xitil', 'EVIVE', 'TROTM', 'IMBEX', 'NIVIL']
suffix_group = ['gents', 'itors', 'sants', 'etics', 'otics',
                'drugs', 'tives', 'lants', 'mines', 'roids']
suffix_drug_n = ['idine', 'PCP', 'gaine', '18-MC', '-NANM',
                 'toxin', 'MHD', 'xin A', 'tatin', 'mPGE2']


def classify_token(txt):
    if txt.lower() in external:
        return external[txt.lower()] 

    elif txt.count('-') >= 3:
        return 'drug_n'

    elif sum(i.isupper() or i in set(['-', '_', ',']) for i in txt) >= 3:
        return "drug_n"

    else:
        return "NONE"


# --------- Entity extractor -----------
# -- Extract drug entities from given text and return them as
# -- a list of dictionaries with keys "offset", "text", and "type"

def extract_entities(stext):

    # WARNING: This function must be extended to
    #          deal with multi-token entities.

    # tokenize text
    tokens = tokenize(stext)

    result = []
    # classify each token and decide whether it is an entity.
    for (token_txt, token_start, token_end) in tokens:
        drug_type = classify_token(token_txt)

        if drug_type != "NONE":
            e = {"offset": str(token_start)+"-"+str(token_end),
                 "text": stext[token_start:token_end+1],
                 "type": drug_type
                 }
            result.append(e)

    return result

# --------- main function -----------


def nerc(datadir, outfile):

    # open file to write results
    outf = open(outfile, 'w')

    # process each file in input directory
    for f in listdir(datadir):

        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)

        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value   # get sentence id
            stext = s.attributes["text"].value   # get sentence text

            # extract entities in text
            entities = extract_entities(stext)

            # print sentence entities in format requested for evaluation
            for e in entities:
                print(sid,
                      e["offset"],
                      e["text"],
                      e["type"],
                      sep="|",
                      file=outf)

    outf.close()


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  baseline-NER.py target-dir
# --
# -- Extracts Drug NE from all XML files in target-dir
# --
# directory with files to process
# datadir = 'data/devel'  # golddir
# outfile = 'out.txt'

datadir = sys.argv[1]
outfile = sys.argv[2]

nerc(datadir, outfile)

evaluator.evaluate("NER", datadir, outfile)

entities_clean = {}
entities, entities_clean, entities_suffix, entities_info = load_gold_NER_ext(
    'data/train')

# e2 = list(entities['CLASS'])
# with open('parsed.txt','w') as file:
#     for entity in e2:
#         file.write(f"{entity}\n")

print('\n' * 4 + '_' * 60)
for key in entities_clean:
    mu = 0
    pct = 0
    for elem in entities_clean[key]:
        mu += len(elem) / len(entities_clean[key])
        if elem[0].isupper():
            pct += 1 / len(entities_clean[key])

    std = 0
    for elem in entities_clean[key]:
        std += 1 / (len(entities_clean[key]) - 1) * (len(elem) - mu)**2

    std = sqrt(std)

    print(key)
    print(f"entities info (numbers): {entities_info[key]['numbers']}")
    print(f"entities info (capital): {entities_info[key]['capital']}")
    print(f"entities info (combo): {entities_info[key]['combo']}")
    print(
        f"entities info (start): {entities_info[key]['start'].most_common(5)}")
    print(
        f"entities info (letters): {entities_info[key]['letters'].most_common(5)}")
    print(f"entities info (dashes): {entities_info[key]['dashes']}")
    print(f"Size of the dataset: {sum(entities_suffix[key].values())}\n")

    print(f'Unique words: {len(entities_clean[key])}')
    print(f'Mean: {round(mu, 2)}')
    print(f'Std: {round(std, 2)}')
    print(f'({mu - std}, {mu + std})')
    print(f'Percentage of uppercase {round(100*pct, 2)}')

    print(entities_suffix[key].most_common(10))

    print('_' * 60)

    # 5, 6 --> brand
    # 7 undecisive
    # 8, 9 --> drug

# drug_n
# Size of the dataset: 23
# Mean: 16.43
# Std: 9.37
# (7.069342819204033, 25.80022239818728)
# ____________________
# drug
# Size of the dataset: 540
# Mean: 11.02
# Std: 3.29
# (7.732574197148425, 14.304462839888613)
# ____________________
# brand
# Size of the dataset: 123
# Mean: 7.98
# Std: 2.75
# (5.228763000039568, 10.738716674757196)
# ____________________
# group
# Size of the dataset: 350
# Mean: 19.04
# Std: 9.28
# (9.756649536027936, 28.317636178257743)
# ____________________

# chmod +x ./startup.sh
