# -------------------
# -- check pattern:  LCS is a verb, one entity is under its "nsubj" and the other under its "obj"

from numpy import negative


def check_LCS_svo(tree, tkE1, tkE2):

    if tkE1 is not None and tkE2 is not None:
        lcs = tree.get_LCS(tkE1, tkE2)

        if tree.get_tag(lcs)[0:2] == "VB":
            path1 = tree.get_up_path(tkE1, lcs)
            path2 = tree.get_up_path(tkE2, lcs)
            func1 = tree.get_rel(path1[-1]) if path1 else None
            func2 = tree.get_rel(path2[-1]) if path2 else None

            if (func1 == 'nsubj' and func2 == 'obj') or (func1 == 'obj' and func2 == 'nsubj'):
                lemma = tree.get_lemma(lcs).lower()

                negative_form = False
                for child in tree.get_children(lcs):
                    if tree.get_lemma(child).lower() == 'not':
                        negative_form = True

                negative_form = False

                if lemma in ['diminish', 'augment', 'exhibit', 'experience', 'counteract', 'potentiate', 'enhance', 'reduce', 'antagonize'] and not negative_form:
                    return 'effect'

                # effect|block_VB : 1.0  --  3 / 3
                # effect|attenuate_VB : 1.0  --  2 / 2
                # effect|reverse_VB : 1.0  --  1 / 1
                # effect|modulate_VB : 1.0  --  1 / 1
                # effect|mask_VB : 1.0  --  1 / 1
                # effect|exacerbate_VB : 1.0  --  1 / 1
                # effect|depress_VB : 1.0  --  1 / 1
                # effect|amplify_VB : 1.0  --  1 / 1
                # effect|accentuate_VB : 1.0  --  1 / 1
                # effect|include_VB : 0.525  --  42 / 80
                # effect|prevent_VB : 0.5  --  3 / 6
                if lemma in ['block', 'attenuate', 'reverse', 'modulate', 'mask', 'exacerbate', 'depress', 'amplify', 'accentuate', 'include', 'prevent'] and not negative_form:
                    return 'effect'

                if lemma in ['impair', 'inhibit', 'displace', 'accelerate', 'bind', 'induce', 'decrease', 'elevate', 'delay'] and not negative_form:
                    return 'mechanism'

                # mechanism|indicate_VB : 1.0  --  3 / 3
                # mechanism|increase_VB : 0.33783783783783783  --  50 / 148
                # mechanism|cause_VB : 0.3333333333333333  --  8 / 24
                # mechanism|modify_VB : 0.3333333333333333  --  1 / 3
                if lemma in ['indicate', 'increase', 'cause', 'modify'] and not negative_form:
                    return 'mechanism'

                if lemma in ['exceed']:
                    return 'advise'

                # advise|maintain_VB : 1.0  --  1 / 1
                if lemma in ['maintain']:
                    return 'advise'

                if lemma in ['suggest'] and not negative_form:
                    return 'int'

    return None

# -------------------
# -- check pattern:  A word in between both entities belongs to certain list


def check_wib(tree, tkE1, tkE2, entities, e1, e2):

    if tkE1 is not None and tkE2 is not None:
        # get actual start/end of both entities
        l1, r1 = entities[e1]['start'], entities[e1]['end']
        l2, r2 = entities[e2]['start'], entities[e2]['end']

        p = []
        for t in range(tkE1+1, tkE2):
            # get token span
            l, r = tree.get_offset_span(t)
            # if the token is in between both entities
            if r1 < l and r < l2:
                lemma = tree.get_lemma(t).lower()

                if lemma in ['tendency', 'stimulate', 'regulate', 'prostate', 'modification', 'augment', 'accentuate', 'exacerbate']:
                    return 'effect'

                # effect|prothrom : 1.0  --  24 / 24
                # effect|bin : 1.0  --  24 / 24
                # effect|adrenocortical : 1.0  --  23 / 23
                # effect|serotoninergic : 1.0  --  15 / 15
                # effect|nimbex : 1.0  --  14 / 14
                # effect|augment : 1.0  --  12 / 12
                # effect|prostate : 1.0  --  8 / 8
                # if lemma in ['prothrom', 'bin', 'adrenocortical', 'serotoninergic', 'nimbex', 'augment', 'prostate']:
                #    return 'effect'

                if lemma in ['react', 'faster', 'presumably', 'induction', 'substantially', 'minimally']:
                    return 'mechanism'

                if lemma in ['exceed', 'extreme', 'cautiously']:
                    return 'advise'

                # advise|tell : 1.0  --  12 / 12
                # advise|methysergide : 1.0  --  12 / 12
                # advise|doctor : 1.0  --  12 / 12
                # advise|nephrotoxic : 1.0  --  9 / 9
                # advise|Solution : 1.0  --  6 / 6
                # advise|Ophthalmic : 1.0  --  6 / 6
                # advise|pure : 0.8571428571428571  --  6 / 7
                # advise|narrow : 0.8333333333333334  --  15 / 18
                # advise|index : 0.8333333333333334  --  10 / 12
                # if lemma in ['tell','methysergide','doctor', 'nephrotoxic', 'Solution', 'Ophthalmic', 'pure', 'narrow', 'index']:
                #    return 'advise'

                if lemma in ['interact']:
                    return 'int'

    return None


# -----------------
# check pattern:  LCS is a verb, and it has a modifier 'should'
def check_should_modifier(tree, tkE1, tkE2):

   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1, tkE2)

      if tree.get_tag(lcs)[0:2] == "VB":

         for c in tree.get_children(lcs):
               if tree.get_lemma(c).lower() == 'should':
                  return 'advise'

   return None


def check_LCS_is_monitor(tree, tkE1, tkE2):

   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1, tkE2)

      if tree.get_tag(lcs)[0:2] == "VB" and tree.get_lemma(lcs) in ['monitor']:
         return 'advise'

   return None


def check_pattern_G3(tree, tkE1, tkE2):
# focused on classes mechanism and effect
   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1, tkE2)

      tag_lcs = tree.get_tag(lcs)
      if tag_lcs[0:2] == "NN":
         if tree.get_lemma(lcs).lower() in ['injection', 'effect']:
            return 'effect'
         elif tree.get_lemma(lcs).lower() in ['example', 'reduction']:
            return 'mechanism'

   return None

def check_LCS_obj(tree, tkE1, tkE2):
   
   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1, tkE2)
      
      for c in tree.get_children(lcs):
         if tree.get_rel(c) == 'obj':
            k = tree.get_lemma(lcs).lower()+'_'+tree.get_lemma(c).lower()
            # effect|increase_response : 1.0  --  54 / 54
            # effect|diminish_response : 1.0  --  24 / 24
            # effect|regulate_proliferation : 1.0  --  8 / 8
            # effect|prolong_time : 1.0  --  4 / 4
            # effect|increase_irritability : 1.0  --  4 / 4
            # effect|contain_epinephrine : 1.0  --  4 / 4
            # effect|cause_arrhythmia : 1.0  --  4 / 4
            # effect|have_consequence : 1.0  --  3 / 3
            # effect|exacerbate_hypertension : 1.0  --  3 / 3
            # effect|evaluate_possibility : 1.0  --  3 / 3
            # effect|produce_effect : 0.9230769230769231  --  12 / 13
            # effect|increase_risk : 0.8333333333333334  --  15 / 18
            # effect|experience_reduction : 0.75  --  3 / 4
            # effect|modify_effect : 0.6666666666666666  --  4 / 6
            # effect|take_capecitabine : 0.6666666666666666  --  2 / 3
            # effect|take_anticoagulant : 0.6666666666666666  --  2 / 3
            if k in ['increase_response', 'diminish_response', 'regulate_proliferation',
                     'prolong_time', 'increase_irritability', 'contain_epinephrine',
                     'cause_arrhythmia', 'have_consequence', 'exacerbate_hypertension',
                     'evaluate_possibility', 'produce_effect', 'increase_risk',
                     'experience_reduction', 'modify_effect', 'take_capecitabine',
                     'take_anticoagulant']:
               return 'effect'
            
            # mechanism|form_complex : 0.8333333333333334  --  5 / 6
            # mechanism|increase_level : 0.75  --  3 / 4
            # mechanism|increase_clearance : 0.75  --  3 / 4
            # mechanism|affect_concentration : 0.6666666666666666  --  4 / 6
            # mechanism|increase_area : 0.6666666666666666  --  2 / 3
            # mechanism|have_which : 1.0  --  3 / 3
            if k in ['form_complex', 'increase_level', 'increase_clearance', 
                     'affect_concentration', 'increase_area', 'have_which']:
               return 'mechanism'
            
            # advise|tell_doctor : 1.0  --  12 / 12
            # advise|exceed_dose : 1.0  --  4 / 4
            # advise|cause_hypermagnesemia : 1.0  --  4 / 4
            if k in ['tell_doctor', 'exceed_dose', 'cause_hypermagnesemia']:
               return 'advise'
               
   return None
                    