#! /usr/bin/python3

from lib2to3.pgen2 import token
from string import punctuation
import sys
import re
from os import listdir

from xml.dom.minidom import parse
from xml.etree.ElementInclude import include
from nltk.tokenize import word_tokenize


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

most_common_suffixes = {
    3: {
        # 'drug': ['ine', 'ide', 'cin', 'ole', 'one'],  # 'ine' is repeated
        'drug': ['ide', 'cin', 'ole', 'one'],
        'brand': ['rin', 'CIN', 'XOL', 'SYS', 'RON'],
        'group': ['nts', 'ics', 'nes', 'ors', 'ids'],
        # 'drug_n': ['ine', 'ate', 'PCP', 'ANM', '-MC']  # 'ine' is repeated
        'drug_n': ['ate', 'PCP', 'ANM', '-MC']
    },

    4: {
        # 'drug': ['dine', 'pine', 'zole', 'mine', 'arin'],  # 'dine' is repeated
        'drug': ['pine', 'zole', 'mine', 'arin'],
        'brand': ['irin', 'OCIN', 'AXOL', 'ASYS', 'IOXX'],
        'group': ['tics', 'ants', 'tors', 'ines', 'ents'],
        # 'drug_n': ['dine', 'NANM', 'aine', '8-MC']  # 'PCP' is present in the 3-suffixes and 'dine' is repeated
        'drug_n': ['NANM', 'aine', '8-MC']
    },

    5: {
        # 'drug': set(['azole', 'amine', 'farin', 'idine', 'mycin']),  # 'idine' is repeated
        'drug': ['azole', 'amine', 'farin', 'mycin'],
        'brand': ['pirin', 'DOCIN', 'TAXOL', 'GASYS', 'VIOXX'],
        'group': ['gents', 'itors', 'sants', 'etics', 'otics'],
        # 'drug_n': ['idine', 'PCP', 'gaine', '18-MC', '-NANM']  # 'PCP' is present in the 3-suffixes and 'idine' is repeated
        'drug_n': ['gaine', '18-MC', '-NANM']
    }
}


# --------- tokenize sentence -----------
# -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    # word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        # keep track of the position where each token should appear, and
        # store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)

    # tks is a list of triples (word, start, end)
    return tks


# --------- get tag -----------
# Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans):
    (form, start, end) = token
    for (spanS, spanE, spanT) in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT

    return "O"

# --------- Feature extractor -----------
# -- Extract features for each token in given sentence


def extract_features(tokens):

    # ----- Best setup -----
    
    # include_punctuation = True
    # include_external_dict = True
    # include_dashes = True
    # inclue_dashes_second = False

    # include_uppercase = False
    # include_all_uppercase = True

    # include_digits = True
    # include_suffix = True
    
    # include_slashes = True
    # include_last_letter = True
    
    # depth = 1

    # M.avg (devel) = 74.2%
    # M.avg (test) = 67.6%
    
    
    # ----- PARAMS -----

    include_punctuation = True
    include_external_dict = True
    include_dashes = True

    include_uppercase = False
    include_all_uppercase = True

    include_digits = True
    include_suffix = True
    
    include_slashes = False
    include_last_letter = False

    depth = 1
    
    # --------------------
    

    # for each token, generate list of features and add it to the result
    result = []
    for k in range(0, len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])

        # tokenFeatures.append("formLower=" + t.lower())
        # tokenFeatures.append("suf4=" + t[-4:])
        # tokenFeatures.append("suf5=" + t[-5:])

        # Ours (External Knowledge)
        if include_external_dict:
            if t.lower() in external:
                tokenFeatures.append("externalDictSays=" + external[t.lower()])

        # Ours
        if include_dashes:
            n_dashes = len(re.findall('-', t))
            tokenFeatures.append("numberOfDashes=" + str(n_dashes))
            

        # Ours
        n_upper = sum(i.isupper() for i in t)
        if include_uppercase:
            if n_upper < 7:
                tokenFeatures.append("numberOfUppercase=" + str(n_upper))
            else:
                tokenFeatures.append("numberOfUppercase>=7")

        if include_all_uppercase:
            if n_upper == len(t):
                tokenFeatures.append("allUppercase")

        # Ours
        if include_digits:
            n_digits = len(re.findall('\d', t))
            if n_digits < 3:
                tokenFeatures.append("numberOfDigits=" + str(n_digits))
            else:
                tokenFeatures.append("numberOfDigits>=3")

        # Ours
        if include_suffix:
            for n_suffix in most_common_suffixes.keys():
                for d_class in ['drug', 'brand', 'group', 'drug_n']:
                    if t[-n_suffix:] in most_common_suffixes[n_suffix][d_class]:
                        tokenFeatures.append(
                            "suffixBelongsToClass_" + d_class + "_WithNoChars=" + str(n_suffix))

        # Ours
        if include_punctuation:
            if t in punctuation:
                tokenFeatures.append("tokenIsPunctuation")
                        
        # Ours
        if include_slashes:
            tokenFeatures.append("/inToken=" + str('/' in t))
         
        # Ours   
        if include_last_letter:
            tokenFeatures.append("lastLetter=" + t[-1])

        for d in range(1, depth + 1):
            if k >= 0 + d:
                tPrev = tokens[k - d][0]
                tokenFeatures.append("depth-" + str(d) +
                                     "_" + "formPrev=" + tPrev)
                tokenFeatures.append("depth-" + str(d) +
                                     "_" + "suf3Prev=" + tPrev[-3:])

                # tokenFeatures.append("formPrevLower=" + tPrev.lower())
                # tokenFeatures.append("suf4Prev=" + tPrev[-4:])
                # tokenFeatures.append("suf5Prev=" + tPrev[-5:])

                # Ours
                if include_punctuation:
                    if tPrev in punctuation:
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "prevTokenIsPunctuation")

                # Ours (External Knowledge)
                if include_external_dict:
                    if tPrev.lower() in external:
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "externalDictSaysPrev=" + external[tPrev.lower()])

                # Ours
                if include_dashes:
                    n_dashes = len(re.findall('-', tPrev))
                    tokenFeatures.append(
                        "depth-" + str(d) + "_" + "numberOfDashesPrev=" + str(n_dashes))

                # Ours
                n_upper = sum(i.isupper() for i in tPrev)
                if include_uppercase:
                    if n_upper < 7:
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "numberOfUppercasePrev=" + str(n_upper))
                    else:
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "numberOfUppercasePrev>=7")

                if include_all_uppercase:
                    if n_upper == len(tPrev):
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "allUppercasePrev")

                # Ours
                if include_digits:
                    n_digits = len(re.findall('\d', tPrev))
                    if n_digits < 3:
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "numberOfDigitsPrev=" + str(n_digits))
                    else:
                        tokenFeatures.append(
                            "depth-" + str(d) + "_" + "numberOfDigitsPrev>=3")

                # Ours
                if include_suffix:
                    for n_suffix in most_common_suffixes.keys():
                        for d_class in ['drug', 'brand', 'group', 'drug_n']:
                            if tPrev[-n_suffix:] in most_common_suffixes[n_suffix][d_class]:
                                tokenFeatures.append(
                                    "depth-" + str(d) + "_" + "prevSuffixBelongsToClass_" + d_class + "_WithNoChars=" + str(n_suffix))
                
                # Ours
                if include_slashes:
                    tokenFeatures.append("depth-" + str(d) + "_" + "/inToken=" + str('/' in tPrev))
                
                # Ours
                if include_last_letter:
                    tokenFeatures.append("depth-" + str(d) + "_" + "lastLetter=" + tPrev[-1])
            elif k == 0:
                tokenFeatures.append("BoS")

            if k <= len(tokens) - 1 - d:
                tNext = tokens[k + d][0]
                tokenFeatures.append("depth+" + str(d) + "_" + "formNext=" + tNext)
                tokenFeatures.append("depth+" + str(d) + "_" +
                                    "suf3Next=" + tNext[-3:])

                # tokenFeatures.append("depth+" + str(d) + "_" + "formNextLower=" + tNext.lower())
                # tokenFeatures.append("depth+" + str(d) + "_" + "suf4Next=" + tNext[-4:])
                # tokenFeatures.append("depth+" + str(d) + "_" + "suf5Next=" + tNext[-5:])

                # Ours
                if include_punctuation:
                    if tNext in punctuation:
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "nextTokenIsPunctuation")

                if include_external_dict:
                    if tNext.lower() in external:
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "externalDictSaysNext=" + external[tNext.lower()])

                # Ours
                if include_dashes:
                    n_dashes = len(re.findall('-', tNext))
                    tokenFeatures.append(
                        "depth+" + str(d) + "_" + "numberOfDashesNext=" + str(n_dashes))

                # Ours
                n_upper = sum(i.isupper() for i in tNext)
                if include_uppercase:
                    if n_upper < 7:
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "numberOfUppercaseNext=" + str(n_upper))
                    else:
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "numberOfUppercaseNext>=7")

                if include_all_uppercase:
                    if n_upper == len(tNext):
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "allUppercaseNext")

                # Ours
                if include_digits:
                    n_digits = len(re.findall('\d', tNext))
                    if n_digits < 3:
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "numberOfDigitsNext=" + str(n_digits))
                    else:
                        tokenFeatures.append(
                            "depth+" + str(d) + "_" + "numberOfDigitsNext>=3")

                # Ours
                if include_suffix:
                    for n_suffix in most_common_suffixes.keys():
                        for d_class in ['drug', 'brand', 'group', 'drug_n']:
                            if tNext[-n_suffix:] in most_common_suffixes[n_suffix][d_class]:
                                tokenFeatures.append(
                                    "depth+" + str(d) + "_" + "nextSuffixBelongsToClass_" + d_class + "_WithNoChars=" + str(n_suffix))
                
                # Ours
                if include_slashes:
                    tokenFeatures.append("depth+" + str(d) + "_" + "/inToken=" + str('/' in tNext))
                
                # Ours
                if include_last_letter:
                    tokenFeatures.append("depth+" + str(d) + "_" + "lastLetter=" + tNext[-1])   
                      
            elif k == len(tokens) - 1:
                tokenFeatures.append("EoS")

        result.append(tokenFeatures)

    return result


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  baseline-NER.py target-dir
# --
# -- Extracts Drug NE from all XML files in target-dir, and writes
# -- them in the output format requested by the evalution programs.
# --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        spans = []
        stext = s.attributes["text"].value   # get sentence text
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start, end) = e.attributes["charOffset"].value.split(
                ";")[0].split("-")
            typ = e.attributes["type"].value
            spans.append((int(start), int(end), typ))

        # convert the sentence to a list of tokens
        tokens = tokenize(stext)
        # extract sentence features
        features = extract_features(tokens)

        # print features in format expected by crfsuite trainer
        for i in range(0, len(tokens)):
            # see if the token is part of an entity
            tag = get_tag(tokens[i], spans)
            print(sid, tokens[i][0], tokens[i][1], tokens[i][2],
                  tag, "\t".join(features[i]), sep='\t')

        # blank line to separate sentences
        print()
