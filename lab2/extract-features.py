#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from util.deptree import *
import patterns

import pickle
import numpy as np
from scipy.sparse import csr_matrix


# -------------------
# -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2):
   feats = set()

   # get head token for each gold entity
   tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
   tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

   if tkE1 is not None and tkE2 is not None:

      # _________ Stage 1 _________
      # features for tokens in between E1 and E2
      for tk in range(tkE1+1, tkE2):
         if not tree.is_stopword(tk):
               word = tree.get_word(tk)
               lemma = tree.get_lemma(tk).lower()
               tag = tree.get_tag(tk)
               feats.add("lib=" + lemma)
               feats.add("wib=" + word)
               feats.add("lpib=" + lemma + "_" + tag)
               feats.add("len_wib_" + word + "=" + str(len(word)))

               # feature indicating the presence of an entity in between E1 and E2
               if tree.is_entity(tk, entities):
                  feats.add("eib")

      # features about paths in the tree
      lcs = tree.get_LCS(tkE1, tkE2)

      path1 = tree.get_up_path(tkE1, lcs)
      path1 = "<".join([tree.get_lemma(x)+"_"+tree.get_rel(x)
                        for x in path1])
      feats.add("path1="+path1)

      path2 = tree.get_down_path(lcs, tkE2)
      path2 = ">".join([tree.get_lemma(x)+"_"+tree.get_rel(x)
                        for x in path2])
      feats.add("path2="+path2)

      path = path1+"<"+tree.get_lemma(lcs)+"_"+tree.get_rel(lcs)+">"+path2
      feats.add("path="+path)

      # _________ Stage 2 _________
      
      p = patterns.check_should_modifier(tree,tkE1,tkE2)
      if p is not None:
         feats.add('pattern_G_act_'+p)

      p = patterns.check_LCS_is_monitor(tree,tkE1,tkE2)
      if p is not None:
         feats.add('pattern_G2_act_'+p)

      p = patterns.check_LCS_svo(tree,tkE1,tkE2)
      if p is not None:
         feats.add('pattern_check_LCS_svo_act_'+p)

      p = patterns.check_wib(tree,tkE1,tkE2,entities,e1,e2)
      if p is not None:
         feats.add('pattern_check_wib_act_'+p)

      p = patterns.check_LCS_obj(tree,tkE1,tkE2)
      if p is not None:
         feats.add('pattern_G4_act_'+p)   
         
      # _________ Stage 3 _________
      word = tree.get_word(lcs)
      lemma = tree.get_lemma(lcs).lower()
      tag = tree.get_tag(lcs)
      feats.add("lemma_lcs=" + lemma)
      feats.add("lcs=" + word)
      feats.add("lcs_tag_" + tag)
      feats.add("len_lcs_" + str(len(lemma)))
      feats.add("number_children_lcs" + str(len(tree.get_children(lcs))))
      
      # _________ Stage 4 _________
      for node in tree.get_nodes():
         w = tree.get_word(node)
         if node < tkE1:
            feats.add(w + "_left")
         elif tkE1 <= node < tkE2:
            feats.add(w + "_middle")
         else:
            feats.add(w + "_right")
            
      # _________ Stage 5 _________
      # features for tokens before E1
      for tk in range(1, tkE1):
         if not tree.is_stopword(tk):
               word = tree.get_word(tk)
               lemma = tree.get_lemma(tk).lower()
               tag = tree.get_tag(tk)
               feats.add("l2L=" + lemma)
               feats.add("w2L=" + word)
               feats.add("lp2L=" + lemma + "_" + tag)
               feats.add("len_w2L_" + word + "=" + str(len(word)))

               # feature indicating the presence of an entity before E1
               if tree.is_entity(tk, entities):
                  feats.add("e2L")
                  
      # features for tokens after E2
      for tk in range(tkE2 + 1, tree.get_n_nodes()):
         if not tree.is_stopword(tk):
               word = tree.get_word(tk)
               lemma = tree.get_lemma(tk).lower()
               tag = tree.get_tag(tk)
               feats.add("l2R=" + lemma)
               feats.add("w2R=" + word)
               feats.add("lp2R=" + lemma + "_" + tag)
               feats.add("len_w2R_" + word + "=" + str(len(word)))

               # feature indicating the presence of an entity after E2
               if tree.is_entity(tk, entities):
                  feats.add("e2R")
      
      # _________ Stage 6 _________
      # features about the paths connecting different entities
      path1_ = tree.get_up_path(tkE1, lcs)
      if len(path1_):
         x = path1_[-1]
         path1_start = tree.get_rel(x)
         path1 = [path1_start]
         for tk in path1_[:-1]:
            if tree.is_entity(tk, entities):
               path1.append(tree.get_rel(tk))            
         path1 = "<".join(path1)
         
         feats.add("path1_G="+path1_start)
         feats.add("path1_G2="+path1)
      else:
         path1_start = ""
         path1 = ""

      path2_ = tree.get_down_path(lcs, tkE2)
      
      if len(path2_):
         x = path2_[0]
         path2_start = tree.get_rel(x)
         path2 = [path2_start]
         for tk in path2_[1:]:
            if tree.is_entity(tk, entities):
               path2.append(tree.get_rel(tk))            
         path2 = ">".join(path2)
         
         feats.add("path2_G="+path2_start)
         feats.add("path2_G2="+path2)
      else:
         path2_start = ""
         path2 = ""

      if path1_start != "" or path2_start != "":
         path_minimal = path1_start+"<"+tree.get_lemma(lcs)+">"+path2_start
         feats.add("path_G2="+path_minimal)
      
      # # Stage 6 (all)
      # if path1 != "" or path2 != "":
      #    path_complete = path1+"<"+tree.get_lemma(lcs)+">"+path2
      #    feats.add("path_G="+path_complete)

      # _________ Stage 7 _________
      root = tree.get_children(0)[0]
      word = tree.get_word(root)
      lemma = tree.get_lemma(root).lower()
      tag = tree.get_tag(root)
      feats.add("lemma_root=" + lemma)
      feats.add("lcs=root" + str(lcs == root))
      feats.add("root_tag" + tag)
      feats.add("len_root_" + str(len(lemma)))
      feats.add("number_children_root" + str(len(tree.get_children(root))))
      
      # # _________ Stage 8 _________
      # clue_verbs = ['diminish', 'augment', 'exhibit', 'experience', 'counteract', 'potentiate', 
      #               'enhance', 'reduce', 'antagonize', 'block', 'attenuate', 'reverse', 'modulate', 
      #               'mask', 'exacerbate', 'depress', 'amplify', 'accentuate', 'include', 'prevent', 
      #               'impair', 'inhibit', 'displace', 'accelerate', 'bind', 'induce', 'decrease', 
      #               'elevate', 'delay',  'indicate', 'increase', 'cause', 'modify', 'exceed', 
      #               'maintain', 'suggest']
      
      # clue_lemmas = ['tendency', 'stimulate', 'regulate', 'prostate', 'modification', 'augment', 
      #                'accentuate', 'exacerbate','react', 'faster', 'presumably', 'induction', 
      #                'substantially', 'minimally', 'exceed', 'extreme', 'cautiously', 'interact']
      
      # for node in tree.get_nodes():
      #    w = tree.get_lemma(node).lower()
      #    if w in clue_verbs or w in clue_lemmas:
      #       if node < tkE1:
      #          feats.add("Left_CV_" + str(w in clue_verbs) + "_CL_" + str(w in clue_lemmas))
      #       elif tkE1 <= node < tkE2:
      #          feats.add("Middle_CV_" + str(w in clue_verbs) + "_CL_" + str(w in clue_lemmas))
      #       else:
      #          feats.add("Right_CV_" + str(w in clue_verbs) + "_CL_" + str(w in clue_lemmas))
               
   return feats


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  extract_features targetdir
# --
# -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
# --

current_id = 0
current_instance = 0
feat_to_id = {}

row_sp = np.array([])
col_sp = np.array([])
data_sp = np.array([])
y_SVM = []

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
            id = e.attributes["id"].value
            offs = e.attributes["charOffset"].value.split("-")
            entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1:
            continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi == "true"):
                dditype = p.attributes["type"].value
            else:
                dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features(analysis, entities, id_e1, id_e2)
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")
            
            for feat in feats:
               if feat not in feat_to_id:
                  feat_to_id[feat] = current_id
                  current_id += 1
            
            ids_feats_to_add = [feat_to_id[feat] for feat in feats]
            col_sp = np.concatenate((col_sp, ids_feats_to_add))
            row_sp = np.concatenate((row_sp, [current_instance]*len(ids_feats_to_add)))
            current_instance += 1
            y_SVM.append(dditype)

data_sp = np.ones(len(row_sp), dtype=int)
row_sp = row_sp.astype(int)
col_sp = col_sp.astype(int)
M_SVM = csr_matrix((data_sp, (row_sp, col_sp)))
y_SVM = np.array(y_SVM)

with open('feat_to_id.pickle', 'wb') as handle:
   pickle.dump(feat_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('M_SVM.pickle', 'wb') as handle:
   pickle.dump(M_SVM, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('y_SVM.pickle', 'wb') as handle:
   pickle.dump(y_SVM, handle, protocol=pickle.HIGHEST_PROTOCOL)
