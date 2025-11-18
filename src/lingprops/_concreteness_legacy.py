# Working on a code that will generate concreteness values for each review
# We have observed that using nouns alone is not enough, as the number of nouns
# does depend on the condition and the specifics of the experiment. Namely,
# the semantic memory reliance means participants will mostly recall relevant
# nouns, thus potentially boosting the concreteness score based on nouns alone.
# We believe including verbs adjectives and adverbs could be essential from the
# theoretical POV (verbs define specific actions, adjectives specify nouns,
# adverbs specify verbs). And also from the computational POV. It seems that
# nouns, verbs and adjectives (some speific POS tags) differ significantly
# between the conditions. Even throughout various datasets.

# The following concreteness parameters TYPES may be of interest:
# categorical vs perceptual concretness;
# total concreteness vs. specific concretness (divided)
# For specific concreteness, we do not want it in SQL, as we can always divide
# by a relevant parameter afterwards as long as we have it in the tables.
# For categorical concretness we have the

# from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
from multiprocessing.pool import Pool

from tqdm import tqdm
import time

# import timing
import numpy as np
np.seterr(all='raise')
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import scipy
from scipy import linalg, optimize, special
from scipy.special import comb
import nltk
# from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag.mapping import tagset_mapping, map_tag
from nltk.stem import WordNetLemmatizer

from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import CorpusReader, WordNetCorpusReader
from nltk.stem import PorterStemmer

from datetime import datetime

old_print = print

def timestamped_print(*args, **kwargs):
  old_print(datetime.now(), *args, **kwargs)

print = timestamped_print

wnl = WordNetLemmatizer()

wn = None    # this global variable will be pointing to different wordnet object for different workers
             # see get_wordnet(),  set_wordnet(), and initializer parameter in Pool

def get_wordnet():
    """ this code is copied from nltk.corpus
    we want wordnet to be individual distinct objects for each worker
    otherwise, if different workers share the same wordnet object
    they confuse each other and the code crashes

    the reason is that wordnet maintains inside it some kind of cache or mutable states

    this function returns a single independent copy of wordnet object
    """

    return LazyCorpusLoader(
    "wordnet",
    WordNetCorpusReader,
    LazyCorpusLoader("omw", CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"))

def set_wordnet():
    """
    this procedure gets a new wordnet object and sets the global variable wn to point to that object
    """
    global wn
    wn = get_wordnet()
    print("initialiazing wordnet: ", wn)

# Functions to generate wordfroms and noun-lemmas they convert to. from wordforms.py
def wordnet_word(word):
    if not wn.synsets(word):
        return False
    else:
        return True

def wordnet_tag(word):
    noun_synsets = wn.synsets(word, pos='n')
    verb_synsets = wn.synsets(word, pos='v')
    adj_synsets = wn.synsets(word, pos='a')
    adv_synsets = wn.synsets(word, pos='r')
    if noun_synsets != [] and verb_synsets == [] and adj_synsets == [] and adv_synsets == []:
        tag = 'NN'
    elif noun_synsets == [] and verb_synsets != [] and adj_synsets == [] and adv_synsets == []:
        tag = 'VB'
    elif noun_synsets == [] and verb_synsets == [] and adj_synsets != [] and adv_synsets == []:
        tag = 'JJ'
    elif noun_synsets == [] and verb_synsets == [] and adj_synsets == [] and adv_synsets != []:
        tag = 'RB'
    else: tag = None
    return tag

def hyphenated_word(word):
    if word[0] != '-' and '-' in word:
        return True
    else:
        return False

def wordent_hyphenation(words):
    # This function takes a list of words and deals with hyphenated words in
    # this list, checking if each of the single words is in WordNet
    i = 0
    while i < len(words): #word in words:
        try: word = words[i]
        except: print("Error in wordent_hyphenation, ", "Index of the word: ", i, "List of words: ", words)
        if hyphenated_word(word):
            if wordnet_word(word):
                i = i + 1
            elif wordnet_word(word.replace('-', '')):
                words[i] = word.replace('-', '')
                i = i + 1
            else: # replace word by wordbeforehyphen, wordafterhyphen
                hyphen_words = word.split('-')
                word1 = hyphen_words[0]
                word2 = hyphen_words[1]
                # print(word, word1, word2)
                words[i] = word1
                words.insert(i+1, word2)
                if len(hyphen_words) > 2:
                    for k in range(2,len(hyphen_words)):
                        words.insert(i+k, hyphen_words[k])
                i = i + len(hyphen_words)
        i = i + 1
    # Need to remove the instances of '' from the list which could appear
    a = 0
    while a < 1:
        try:
            words.remove('')
        except:
            a = 1
    return words

def derivational_conversion(word, from_pos, to_pos):
    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in ('a', 's') and s.name().split('.')[1] in ('a', 's'):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in ('a', 's') and l.synset().name().split('.')[1] in ('a', 's'):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    # print(words)
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    used = set()
    unique_list = [x for x in words if x not in used and (used.add(x) or True)]
    # unique_list = set(words) # added this line for testing and edited it in the follwoing line  as well
    # print(unique_list) # to check if the order stays the same
    result = [(w, float(words.count(w)) / len_words) for w in unique_list]
    result.sort(key=lambda w:-w[1]) # result = sorted(result, key=lambda w:-w[1])# Changed the original to keep the order intact over re-runs of the the code. The original version: result.sort(key=lambda w:-w[1])

    return result

def nounalike_conversion(word, from_pos):
    """ This function checks if there is an identically spelled noun in WordNet"""
    synsets = wn.synsets(word, from_pos)

    # Word not found
    if not synsets:
        return []

    # for s in synsets:

    syn_1 = synsets[0]
    word_1 = syn_1.name().split('.')[0]
    noun_synset_1 = wn.synsets(word_1, pos='n')
    if noun_synset_1 != []:
        result = [(word_1, 1)]
    else:
        return []

    return result

def attribute_conversion(word, from_pos):
    """ This function converts a word to a noun using the attribute method from WordNet"""
# The attribute method exists for adjectives I think
    synsets = wn.synsets(word, from_pos)

    # Word not found
    if not synsets:
        return []

    result =[]
    attribute_list = []
    for s in synsets:
        # word_g = s.name().split('.')[0]
        attrib_s = s.attributes()
        if len(attrib_s) > 1:
            print('There is more than 1 attribute: ', s, attrib_s)
        attribute_list += attrib_s
    for a in attribute_list:
        word_a = a.name().split('.')[0]
        noun_a = wn.synsets(word_a, pos='n')
        if noun_a != []:
            result = [(word_a, 1)]
            break
        else: continue

    return result

def convert_similartos(word, from_pos):
    """ Transforms words uing synomyms (similar_tos) method from WordNet"""
    synsets = wn.synsets(word, from_pos)

    # Word not found
    if not synsets:
        return []

    synsets_similar = []
    for s in synsets:
        similar_s = s.similar_tos() # gives a list of synsets similar ot this one
        synsets_similar += similar_s
    # if not synsets_similar:
    #     result = []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets_similar:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in ('a', 's') and s.name().split('.')[1] in ('a', 's'):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == 'n':
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    # print(words)
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    used = set()
    unique_list = [x for x in words if x not in used and (used.add(x) or True)]
    # unique_list = set(words) # added this line for testing and edited it in the follwoing line  as well
    # print(unique_list) # to check if the order stays the same
    result = [(w, float(words.count(w)) / len_words) for w in unique_list]
    result.sort(key=lambda w:-w[1]) # result = sorted(result, key=lambda w:-w[1])# Changed the original to keep the order intact over re-runs of the the code. The original version: result.sort(key=lambda w:-w[1])

    return result # the result is a list of tuples with (word, word-frequency) as a tuple

def convert_pertainym(word):
    """ Transforms adverbs into adjectives"""
    synsets = wn.synsets(word, 'r')

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            lemmas += [l]

    # Get pertainyms
    pertainyms = [(l, l.pertainyms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_adj_lemmas = []

    for prt in pertainyms:
        for l in prt[1]:
            if l.synset().name().split('.')[1] in ['a', 's']:
                related_adj_lemmas += [l]
            else:
                print('Pertainym for the word is not an adjectif: ', word, l.synset().name().split('.')[1])

    # Extract the words from the lemmas
    words = [l.name() for l in related_adj_lemmas]
    # print(words)
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    used = set()
    unique_list = [x for x in words if x not in used and (used.add(x) or True)]
    # unique_list = set(words) # added this line for testing and edited it in the follwoing line  as well
    # print(unique_list) # to check if the order stays the same
    result = [(w, float(words.count(w)) / len_words) for w in unique_list]
    result.sort(key=lambda w:-w[1]) # result = sorted(result, key=lambda w:-w[1])# Changed the original to keep the order intact over re-runs of the the code. The original version: result.sort(key=lambda w:-w[1])

    return result # the result is a list of tuples with (word, word-frequency) as a tuple

def convert_to_noun(word, from_pos):
    """ Transform words given from/to POS tags """

    if word.lower() in ['most', 'more'] and from_pos == 'a':
        word = 'many'

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    result = derivational_conversion(word, from_pos, 'n')

    if len(result) == 0:
        result = attribute_conversion(word, from_pos)

    if len(result) == 0 and word[-2:].lower() == 'ed' and from_pos != 'v':
        result = derivational_conversion(word, 'v', 'n')

    if len(result) == 0:
        result = convert_similartos(word, from_pos)

    if len(result) == 0 and from_pos == 'r': # working with pertainyms
        adj_words = convert_pertainym(word)
        for adj in adj_words:
            word_a = adj[0]
            # print(word_a)
            result = derivational_conversion(word_a, 'a', 'n')
            if len(result) == 0:
                result = attribute_conversion(word_a, 'a')
            else: break
            if len(result) == 0 and word_a[-2:].lower() == 'ed' and from_pos != 'v':
                result = derivational_conversion(word_a, 'v', 'n')
            else: break
            if len(result) == 0:
                result = convert_similartos(word_a, 'a')
            else: break

    if len(result) == 0:
        result = nounalike_conversion(word, from_pos)

    # return all the possibilities sorted by probability
    return result

def nounify(word, tag):
    noun_list = convert_to_noun(word, tag)
    if noun_list != []:
        noun = noun_list[0][0]
    else:
        # print('Not found in derivationally related forms: ', word, tag)
        if word == 'visuals': noun = 'picture'
        elif word.lower() == 'gameplay': noun = 'game'
        elif word.lower() == 'personalization': noun = 'individualization'
        elif word.lower() == 'coworker': noun = 'co-worker'
        elif word.lower() == 'coworkers': noun = 'co-workers'
        elif word.lower() == 'sans': noun = 'font'
        elif word.lower() == 'microsoft': noun = 'corporation'
        elif word.lower() == 'ios': noun = 'software'
        elif word.lower() == 'powerpoint': noun = 'programme'
        elif word.lower() == 'youtube': noun = 'website'
        elif word.lower() == 'hodge': noun = 'surname'
        elif tag == 'n' and 'thing' in word.lower():
            noun = 'thing'
        elif tag[0] == 'n' and word.lower() in ['anyone', 'everyone', 'anybody', 'everybody']:
            noun = 'person'
        elif tag == 'a':
            noun_list = convert_to_noun(word, 'v')
            if noun_list != []:
                noun = noun_list[0][0]
            else: noun = None
        else: noun = None

    return noun

def wordformtion(text):
    # this function generates 2 dictionaries: a dictionary of wordforms with POS and count
    # and a dictionary of corrsponding noun-lemmas with counts

    porter = PorterStemmer()
    sentences = sent_tokenize(text)
    wordforms = dict() # Initializes an empty dictionary where we will keep track of all nouns in the whole corpus of reviews and how many times their occurence values
    word_wordforms = dict() # The wordforms whithout punctuation, CC, DT, EX, IN

    for sentence in sentences:
        words = word_tokenize(sentence)
        indexes_todelete = [] # A list of indicies of the wrds to be deleted (artifacts from word_tokenization)
        for i in range(1, len(words)):
            word = words[i]
            word.strip() # Remove the whitespaces from the beginning and the end of the word added on 20210127
            words[i] = word
            if words[i-1][0] == "'" and words[i] == "'":
                # print('A word originally in single quatation marks, before the first quatation mark removed: ', words[i-1])
                words[i-1] = words[i-1][1:]
                indexes_todelete = indexes_todelete + [i]
        words = np.delete(words, indexes_todelete).tolist()
        words = list(filter(None, words)) #added on 20210127 to remove empty words
        for i in range(1, len(words)):
            word = words[i]
            try:
                if word[0] == "'" and word[-1] == "'" and word not in ["'", "''"]:
                    word = word[1:-2] # Added and word != "'" on 20210126
                    word.strip()
                    words[i] = word
                elif word[0] == "'" and word != "'":
                    word = word[1:]
                    word.strip()
                    words[i] = word
                elif word[0] == '-' and word != '-':
                    word = word[1:]
                    word.strip()
                    words[i] = word
                elif word in ["'", "''"]: words[i] = None
                elif word != None:
                    word.strip()
                    words[i] = word
            except:
                word.strip()
                words[i] = word
                if len(word) > 0:
                    print('The code block that removes the apostrophe artifacts break on the word: ', word)
                continue
            # if len(words[i]) == 0:
            #     print(word, sentence)
        words = list(filter(None, words))

        # Here we need to insert treatement of hyphenated words
        words = wordent_hyphenation(words)

        try:
            nltk_tagged = nltk.pos_tag(words)
        except:
            print('NLTK-tagging fails on the following sentence: ', words)
            continue

        a = 0 # setting the marker for the preceeding word being a verb
        for word, tag in nltk_tagged:
# The next piece deals with corrections to POS tagging
            if word.lower() in ['sans', 'microsoft', 'powerpoint', 'youtube', 'ios']:
                tag = 'NNP'
            elif word.lower() in ['app', 'pt', 'neck', 'bottom', 'font', 'kind', 'stiff', 'collar']:
                tag = 'NN'
            elif word.lower() in ['apps', 'thumbs', 'drawbacks']:
                tag = 'NNS'
            elif word.lower() in ['wow', 'aw']:
                tag = 'UH'
            elif tag == 'NNP' and word.lower() in ['boy']:
                tag = 'UH'
            elif word.lower() in ['weird', 'overall', 'great', 'ok', 'stupid', 'okay', 'perfect', 'ok.', 'full']:
                tag = 'JJ'
            elif tag[:2] == 'VB' and word.lower() in ['potential', 'routine', 'ping']:
                tag = 'NN'
            elif tag[:2] == 'VB' and word.lower() in ['deep', 'straight', 'simple', 'stiff', 'groundbreaking', 'good', 'handy', 'specific', 'daily', 'glad', 'sore', 'quick', 'sobering', 'fun']:
                tag = 'JJ'
            elif tag[:2] == 'VB' and word.lower() in ['more', 'sideways']:
                tag = 'RB'
            elif tag[:2] == 'JJ' and word.lower() in ['web']:
                tag = 'NN'
            elif tag[:2] == 'JJ' and word.lower() in ['aside']:
                tag = 'RB'
            elif tag[:2] == 'RB' and word.lower() in ['silly', 'friendly', 'sore', 'nice']:
                tag = 'JJ'
            elif tag[:2] == 'RB' and word.lower() in ['neck', 'strain', 'winter', 'pain', 'flows']:
                tag = 'NN'
            elif tag[:2] == 'NN' and word.lower() in ['begin', 'do', 'clasp', 'say']:
                tag = 'VB'
            elif tag == 'NNS' and word.lower() in ['uses', 'teaches', 'eases']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'VBZ'
            elif tag[0] == 'N' and word.lower() in ['saved', 'developed']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'VBD'
            elif tag[0] == 'N' and word.lower() in ['described']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'VBN'
            elif tag[0] == 'N' and word.lower() in ['buzzing', 'matching', 'crashing', 'staring']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'VBG'
            elif tag[0] == 'N' and word.lower() in ['soothing', 'condescending', 'entertaining', 'amazing', 'relaxing', 'challenging', 'interesting', 'confusing', 'damaging', 'nagging', 'changing', 'decent', 'easy', 'slow', 'relaxed', 'sure', 'goofy', 'quick']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'JJ'
            elif tag[0] == 'N' and word.lower() in ['quicker']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'JJR'
            elif tag[0] == 'N' and word.lower() in ['pretty', 'anytime', 'forth', 'first']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'RB'
            elif tag[0] == 'N' and word.lower() in ['towards', 'about']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'IN'
            elif tag[0] in ['N', 'V'] and word.lower() in ['ourselves', 'myself']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'PRP'
            elif tag[0] == 'N' and word.lower() in ['yours']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'PRP$'
            elif tag[0] == 'V' and word.lower() in ['everything']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'NN'
            elif tag[0] == 'V' and word.lower() in ['easy', 'tight']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'JJ'
            elif tag[0] == 'V' and word.lower() in ['that']: # or word.lower() == 'coherent' or word.lower() == 'helpful'):
                tag = 'PR'
            elif word.lower() == 'alright':
                tag = 'RB'
            elif tag[:2] != 'NN' and word.lower() in ['neck']:# , 'workday', 'workplace', 'desk']:
                tag = 'NN'
            elif len(word) > 2 and wordnet_tag(word) != None and (tag[:2] != wordnet_tag(word) or word == 'font') and (wordnet_tag(word) != 'NN' or 'font' in word): # and tag[0] in ['N', 'V', 'J', 'R'] I have removed the NN tagged words, as they often overlap with some strange words, abbreviations for pronouns or propositions which are not in wordnet. With the exclusion of the word font
                # print('Before tag replacement: ', word, tag)
                tag = wordnet_tag(word)
            elif len(word) > 2 and wordnet_tag(word) != None and (tag[:2] != wordnet_tag(word) or word.lower() == 'fun') and wordnet_tag(word) == 'NN' and word not in ['might']: # and tag[0] in ['N', 'V', 'J', 'R'] I have removed the NN tagged words, as they often overlap with some strange words, abbreviations for pronouns or propositions which are not in wordnet. With the exclusion of the word font
                if word.lower() not in ['why', 'its', 'who', 'may', 'yes', 'tho', 'while', 'otter', 'upside', 'genius', 'despite', 'sceptic', 'lifesaving']: # Note: removed the word 'fun' from the exclusion list
                    # print('Retagging as a noun. Before tag replacement: ', word, tag)
                    # print(sentence)
                    tag = wordnet_tag(word)

            if a >= 1 and tag[0] == 'V': # handling auxiliary verbs
                wordforms[(word_prev, tag_prev)] -= 1
                if wordforms[(word_prev, tag_prev)] < 0: # added on 20210127. Negative values do not make sence and lead to errors in calculating concreteness on verbs
                    # wordforms[(word_prev, tag_prev)] = 0
                    print("Error. The wordform count becomes negative: ", "For worform: ", word_prev, tag_prev, "On wordform: ", word, tag, "On sentence: ", sentence)

                wordforms[(word_prev, 'AU')] = wordforms.get((word_prev, 'AU'), 0) + 1
                a = 0 # reset the counter after tagging the AU verb, so that we do not count it again

            # if word.lower() == 'dr':
            #     print(sentence)
            if tag[0] == 'V' and word.lower() in ['am', "m", "'m", 'is', "s", "'s", 'are', "re", "'re", 'was', 'were', 'being', 'been', 'be', 'have', "ve", "'ve", 'has', 'had', 'd', "'d", 'do', 'does', 'did', 'will', 'll', "'ll", 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could']:
                a += 1 # a marker to memorize that the word was a verb to handle auxiliary verbs
                if a > 3:
                    # print('More than 3 consecutive verbs detected in the following sentence: ', nltk_tagged)
                    a = 0
                    print("Sentence on which AU tagging breaks: more than 3 verbs in a row: ",sentence)

                tag_prev = tag
                word_prev = word.lower()
            elif a >= 1 and (tag[:2] == 'RB' or word.lower() in ['not' , "n't", 't', "'t"]):
                a += 1
                # if word.lower() in ["n't", 't', "'t"]:
                #     print("Sentence containing n't, t or 't'", nltk_tagged)
            else: a = 0

            wordforms[(word.lower(), tag)] = wordforms.get((word.lower(), tag), 0) + 1

    return wordforms

def noun_lemmas(wordforms):
    """This function receves as input a dictionary of wordforms and outputs the corresponding noun-lemmas as a dictionary with wordform(word, tag) as key and the noun-lemma as the value"""

    all_nouns = dict()
    wordforms_notinWordNet = []
    for w in wordforms:
        word = w[0]
        tag = w[1]
            # Now let's update the list of nouns.
            # First, we ensure that the word quaifies. That is: 1) it is longer than 2 characters
        if tag[:2] == 'VB' and word == 're':
            word = 'are'
        elif tag[:2] == 'VB' and word == 've':
            word = 'have'

        if ((len(word) > 2 or tag == 'CD' or (tag != 'AU' and word in ['be', 'do', 'go', 'ad', 'is', 'am'])) and word != "n't") or (tag[:2] == 'NN' and word.lower() in ['pc', 'pt', 'ms']): # and tag in ['N', 'V', 'J', 'R']
            if word in ['app', 'apps']:
                word_rep = 'application'
            # elif tag == 'NN' and word.lower() in ['pc']: # ;wnl.lemmatize doesn't work on double words
            #     word_rep = 'personal computer'
            #     print(word_rep)
            elif tag[:2] == 'NN' and word in ['pt']: # ;wnl.lemmatize doesn't work on double words
                word_rep = 'therapist'
            elif tag == 'NNP' and word.lower() in ['ms']: # ;wnl.lemmatize doesn't work on double words
                word_rep = 'microsoft'
            elif tag[:2] == 'JJ' and word in ['ok', 'ok.']: # ;wnl.lemmatize doesn't work on double words
                word_rep = 'satisfactoriness'
            elif word in ['ios']: # ;wnl.lemmatize doesn't work on double words
                word_rep = 'software'
            elif 'smartphone' in word:
                word_rep = 'phone'
            elif tag == 'NNP' and word == 'kevin':
                word_rep = 'person'
            elif tag[0] == 'N' and word in ['others']:
                word_rep = 'people'
            elif 'redesign' in word:
                word_rep = 'design'
            elif 'restructure' in word:
                word_rep = 'structure'
            elif 'realign' in word:
                word_rep = 'align'
            elif tag[0] == 'N' and word == 'rhyming':
                word_rep = 'rhyme'
            elif 'download' in word:
                word_rep = 'transfer'
            elif 'customize' in word:
                word_rep = 'custom'
            elif 'thank' in word:
                word_rep = 'thanks'
            elif 'keyboarding' in word:
                word_rep = 'keyboard'
            elif 'multitasking' in word:
                word_rep = 'task'
            elif 'off-putting' in word:
                word_rep = 'appeal'
            elif 'inexcusable' in word:
                word_rep = 'excuse'
            elif tag[:2] == 'VB' and word == 'due':
                word_rep = 'do'
            elif tag[0] == 'V' and 'enable' in word:
                word_rep = 'ability'
            # elif tag[0] == 'V' and word == 'sobering':
            #     word_rep = 'sobriety'
            elif tag[0] == 'J' and word == 'unorganized':
                word_rep = 'organization'
            elif tag[0] == 'J' and word == 'hypermobile':
                word_rep = 'mobility'
            elif tag[0] == 'J' and word == 'memorable':
                word_rep = 'memory'
            elif tag[0] == 'J' and word == 'delightful':
                word_rep = 'delight'
            elif tag[0] == 'J' and word == 'optional':
                word_rep = 'option'
            elif tag[0] == 'J' and word == 'outdated':
                word_rep = 'date'
            elif tag[0] == 'J' and word == 'positional':
                word_rep = 'position'
            elif tag[0] == 'J' and word == 'unfocused':
                word_rep = 'focus'
            elif tag[0] == 'J' and word == 'descriptive':
                word_rep = 'description'
            elif word in ['never', 'once', 'already', 'full-time', 'ever', 'initially', 'again', 'sometimes', 'before', 'yet', 'soon', 'ahead', 'anytime', 'eventually', 'finally', 'ago', 'throughout']:
                word_rep = 'time'
            elif tag[:2] == 'RB' and word in ['prior']:
                word_rep = 'time'
            elif word in ['maybe', 'perhaps']:
                word_rep = 'possibility'
            elif tag == 'RB' and word in ['quite', 'bit', 'far']:
                word_rep = 'extent'
            elif tag == 'RB' and word in ['long']:
                word_rep = 'length'
            elif tag[0] == 'R' and word == 'simply':
                word_rep = 'simplicity'
            elif tag[0] == 'R' and word == 'professionally':
                word_rep = 'profession'
            elif tag[0] == 'R' and word == 'supposedly':
                word_rep = 'supposition'
            elif tag[0] == 'R' and word == 'undoubtedly':
                word_rep = 'doubt'
            elif tag[0] == 'R' and word == 'continually':
                word_rep = 'continuity'
            elif tag[0] == 'R' and word == 'safely':
                word_rep = 'safety'
            elif tag[0] == 'R' and word == 'routinely':
                word_rep = 'routine'
            elif tag[0] == 'R' and word == 'additionally':
                word_rep = 'addition'
            elif tag[0] == 'R' and word == 'namely':
                word_rep = 'name'
            elif tag[0] == 'R' and word == 'periodically':
                word_rep = 'period'
            elif tag[0] == 'R' and word == 'relaxed':
                word_rep = 'relaxation'
            elif word in ['another', 'every', 'both', 'either', 'together', 'anymore', 'almost', 'else']:
                word_rep = 'number'
            elif word in ['visually']:
                word_rep = 'vision'
            elif tag[0] == 'R' and word in ['most', 'more']:
                word_rep = 'group'
            elif tag[0] == 'R' and word in ['around', 'away', 'elsewhere', 'wherever', 'anywhere', 'between', 'sidewards', 'forth']:
                word_rep = 'place'
            elif tag[0] == 'R' and word in ['loose']:
                word_rep = 'looseness'
            elif tag[:2] == 'RB' and word in ['lighter']:
                word_rep = 'lightness'
            else:
                word_rep = word
            noun = None # pre-assign the variable noun to None
            # check if the word is found in WordNet as it is:
            if (tag[0] == 'N' or tag == 'CD') and wn.synsets(wnl.lemmatize(word_rep,'n'), pos='n') != []:
                noun = wnl.lemmatize(word_rep,'n') # = all_nouns.get((word.lower(), tag, wnl.lemmatize(word_rep,'n')), 0) + 1
            elif 'sideway' in word_rep:
                noun = ['side', 'way'] # = all_nouns.get((word.lower(), tag, ('side', 'way')), 0) + 1
            # elif tag[0] == 'N' and word.lower() == 'rhyming':
            #     all_nouns['rhyme'] = all_nouns.get('rhyme', 0) + 1
            elif tag[0] in ['N', 'V', 'J', 'R'] and tag != 'RP': # Added on 20200520 "and tag != 'RP'" to exclude Particles. New idea: use derivationally related forms etc. Original idea: Transform the word through stemming and lemmatization
                short_tag = tag[0].lower() # generate a short-tag from POS tag
                if short_tag == 'j': short_tag = 'a'

                noun = nounify(word_rep, short_tag) # prints out word and short_tag if not found in Wordnet

                if noun == None and word_rep not in ['also', 'not', 'just', 'too', 'instead', 'only', 'very', 'rather', 'however', 'esque', 'but', 'anyway', 'furthermore', 'about', 'though', 'regardless', 'alright', 'further', 'mostly', 'anyways', 'nonetheless', 'virtually', 'beyond', 'along', 'alongside', 'somehow']:# and word.lower()[-2:] != 'ly':
                    # check if the word is found in WordNet as it is:
                    if wn.synsets(wnl.lemmatize(word_rep,'n'), pos='n') != [] and word not in ['tho', 'otter']:
                        noun = wnl.lemmatize(word_rep,'n') # = all_nouns.get((word.lower(), tag, wnl.lemmatize(word_rep,'n')), 0) + 1

            if tag[:2] in ['NN', 'VB', 'JJ', 'RB', 'CD'] and noun == None and word_rep not in ['also', 'not', 'just', 'too', 'instead', 'only', 'very', 'rather', 'however', 'esque', 'but', 'anyway', 'furthermore', 'about', 'though', 'regardless', 'alright', 'further', 'mostly', 'anyways', 'nonetheless', 'virtually', 'beyond', 'along', 'alongside', 'somehow', 'thus']:# and word.lower()[-2:] != 'ly':
                wordforms_notinWordNet = wordforms_notinWordNet + [w]
            elif noun != None:
                    all_nouns[w] = noun # = all_nouns.get((word.lower(), tag, noun), 0) + 1

    return all_nouns, wordforms_notinWordNet

# Now we have a dictionary with nouns and their occurences. We could replace it
# with a dictionary of nouns excluding stopwords or we could exclude stopwords
# when we calculate the depth of the review only

# Now lets define the fuctions to find both hypernym and hyponym depth.
def hypernym_depth(word, postag):
    return wn.synsets(wnl.lemmatize(word, postag), postag)[0].min_depth() #this selects the first synset. We could think of a smarter way of selecting a synset

def get_hypernyms(synset):
    hypernyms = set()
    for h in synset.hypernyms():
        hypernyms |= set(get_hypernyms(h))
    return hypernyms | set(synset.hypernyms())

def hyp_num(noun, POS): # 20200617 added POS option to make sure we pick the right synset (proper none or not)
    synsets = wn.synsets(noun, 'n')
    if POS != 'NNP':
        for s in synsets:
            if s.instance_hypernyms() != []:
                continue
            else:
                synset = s
                break
    else:
        synset = synsets[0]
        for s in synsets:
            if s.instance_hypernyms() == []:
                continue
            else:
                synset = s.instance_hypernyms()[0]
                break
    # first_s.name().split('.')[1]
    hyp_set = get_hypernyms(synset)
    if POS == 'NNP':
        return 1 + len(hyp_set)
    else:
        return len(hyp_set)

def text_depth(text, postag, repetitions, exclusion_list):
    # text is is the input uncleaned text for which to calculate depth
    # postag - pos tag included in the count
    # repetitions - true (include repetitions in the counts), false ( count only once) what to do if the same wordform is repeated
    # output is the total concreteness and the number of words used in the calculation

    # 20200526: 2 options input text or input the all_nouns object, which is a dictionary that relates the key (worform, pos) to the value (noun it transforms into)
    from scipy.special import comb #scipy is a mathematical packet and comb is the function we need to calculate the entropy
    # number = 1 # initiating the number of texts which are as deep or more general than the given text
    conc = 0

    # Convert the text into the noun-wordforms dictionary:
    word_forms = wordformtion(text)
    [nouns, wordforms_notinWordNet] = noun_lemmas(word_forms)

    number_nouns = 0 # initiating the count of nouns in the review

    if repetitions:
        for wordform in word_forms:
            try:
                noun = nouns[wordform] # Note!: noun can be a list of nouns
            except:
                continue
            if noun in exclusion_list: continue
            elif wordform[1][:2] not in postag: continue
            else:
                if type(noun) == list:
                    for n in noun:
                        depth_n = hyp_num(n, wordform[1])
                        if (len(n)<2 or (depth_n == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                        else:
                            frequency = word_forms[wordform] #all_nouns is a dictionary in which the keys are the nouns and their values are their frequencies
                            # number = number * comb(depth_n + 1 + frequency - 1, frequency) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                            conc += np.log(comb(depth_n + 1 + frequency - 1, frequency)) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                            # if number < 1:
                            #     print("Number less than 1: ", "on noun: ", noun, "noun-part: ", n, "POS tag: ", wordform[1], "frequency: ", frequency, "Number: ", number)
                            if conc < 0:
                                print("Concreteness less than 0: ", "on noun: ", noun, "POS tag: ", wordform[1], "frequency: ", frequency, "Concreteness: ", conc)

                    number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
                    # print(' rep Nouns in lsts give: ', number_nouns, frequency)
                else:
                    try:
                        depth_noun = hyp_num(noun, wordform[1])
                        if wordform[0] in ['gameplay', 'ios', 'pt'] or 'smartophone' in wordform[0]:
                            depth_noun = depth_noun + 1
                        # if wordform[0] in ['gameplay', 'sans', 'microsoft', 'powerpoint', 'youtube', 'hodge', 'pt'] or 'smartophone' in wordform[0]:
                        #     depth_noun = depth_noun + 1
                        # elif wordform[0] in ['ios']:
                        #     depth_noun = depth_noun + 2
                    except:
                        # print('rep Fails to calculate depth: ', noun, wordform[1], nouns)
                        continue # wqs break. fixed 20200729
                    if (len(noun) < 2 or (depth_noun == 0 and noun != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                    else:
                        frequency = word_forms[wordform] #all_nouns is a dictionary in which the keys are the nouns and their values are their frequencies
                        number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
                        # number = number * comb(depth_noun + 1 + frequency - 1, frequency) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                        conc += np.log(comb(depth_noun + 1 + frequency - 1, frequency)) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                        # if number < 1:
                        #     print("Number less than 1: ", "on noun: ", noun, "POS tag: ", wordform[1], "frequency: ", frequency, "Number: ", number)
                        if conc < 0:
                            print("Concreteness less than 0: ", "on noun: ", noun, "POS tag: ", wordform[1], "frequency: ", frequency, "Concreteness: ", conc)
                        # print('rep Noun ', noun, 'gives ', number_nouns, frequency)
    else:
        # Need to create a new nouns subdictionary, which only contains the proper POS tags
        nouns_tag = dict()
        for w, n in nouns.items():
            if w[1][:2] not in postag: continue
            elif n in exclusion_list: continue
            else:
                nouns_tag[w] = n
        # Create a list/set of unique values:
        unique_nouns_tag = []
        for n in nouns_tag.values():
            if n not in unique_nouns_tag:
                unique_nouns_tag.append(n)
        for noun in unique_nouns_tag: # set(nouns_tag.values()):
            # The following commented section calculates the frequency of noun-lemmas (that is it makes equivalent different wordforms which give same noun-lemma - allows for permutations amonfg the distinct wordforms)
            frequency = 1
            # number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review

            if type(noun) == list:
                for n in noun:
                    try:
                        depth_n = hyp_num(n, 'NN')
                    except:
                        depth_n = hyp_num(n, 'NNP')
                    if (len(n)<2 or (depth_n == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                    else:
                        # number = number * comb(depth_n + 1 + frequency - 1, frequency) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                        conc += np.log(comb(depth_n + 1 + frequency - 1, frequency)) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
                # print('norep Nouns in lsts give: ', number_nouns, frequency)
            else:
                try:
                    depth_noun = hyp_num(noun, 'NN')
                except:
                    depth_noun = hyp_num(noun, 'NNP')
                # create a list of wordforms which convert into a given noun:
                wordf_n = [w[0] for w, n in nouns.items() if  n == noun]
                if len(set(wordf_n) & set(['gameplay', 'sans', 'microsoft', 'powerpoint', 'youtube', 'hodge', 'pt', 'smartphone', 'smartphones'])) > 0:
                    depth_noun = depth_noun + 1
                elif len(set(wordf_n) & set(['ios'])) > 0:
                    depth_noun = depth_noun + 2
                if (len(noun)<2 or (depth_noun == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                else:
                    # number = number * comb(depth_noun + 1 + frequency - 1, frequency) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                    conc += np.log(comb(depth_noun + 1 + frequency - 1, frequency)) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                    number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
                    # print('norep Noun ', noun, 'gives ', number_nouns, frequency)

# The following is the original version of the code which treats nonrepetitions per wordform rather than per noun
    # for wordform in word_forms:
    #     if wordform[0] in exclusion_list: continue
    #     elif wordform[1][:2] not in postag: continue
    #     else:
    #         try:
    #             noun = nouns[wordform] # Note!: noun can be a list of nouns
    #         except:
    #             continue
    #         if type(noun) == list:
    #             for n in noun:
    #                 depth_n = hypernym_depth(n, 'n')
    #                 if (len(n)<2 or (depth_n == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
    #                 else:
    #                     if repetitions:
    #                         frequency = word_forms[wordform] #all_nouns is a dictionary in which the keys are the nouns and their values are their frequencies
    #                     else:
    #                         frequency = 1
    #                     number = number * comb(depth_n + 1 + frequency - 1, frequency) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
    #             number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
    #         else:
    #             try:
    #                 depth_noun = hypernym_depth(noun, 'n')
    #                 if wordform[0] in ['gameplay', 'sans', 'microsoft', 'powerpoint', 'youtube', 'hodge', 'pt'] or 'smartophone' in wordform[0]:
    #                     depth_noun = depth_noun + 1
    #                 elif wordform[0] in ['ios']:
    #                     depth_noun = depth_noun + 2
    #             except:
    #                 print(noun, nouns)
    #                 break
    #             if (len(noun) < 2 or (depth_noun == 0 and noun != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
    #             else:
    #                 if repetitions:
    #                     frequency = word_forms[wordform] #all_nouns is a dictionary in which the keys are the nouns and their values are their frequencies
    #                 else:
    #                     frequency = 1
    #                 number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
    #                 number = number * comb(depth_noun + 1 + frequency - 1, frequency) # Note depth_noun needs to be bigger by 1, but 1 is being subtracted

    # if number > 0:
    #     try:
    #         return np.log(number), number_nouns
    #     except:
    #         print("Concreteness calculation failed on number: ", number)
    #         print("Comment: ", text)
    # else:
    #     print("Concretenes calculation error on: ", postag, "repetitions: ", repetitions, "Number: ", number)
    #     print("Comment: ", text)

    try:
        return conc, number_nouns
    except:
        print("Concretenes calculation error on: ", postag, "repetitions: ", repetitions, "Number: ", conc)

def text_depth_lin(text, postag, repetitions, exclusion_list): # Note, we do not use this function in this code. Otherwise it could be ncecesary to replace the log of a product by the sum of logs
    # text is is the input uncleaned text for which to calculate depth
    # postag - pos tag included in the count
    # repetitions - true (include repetitions in the counts), false ( count only once) what to do if the same wordform is repeated
    # output is the total concreteness and the number of words used in the calculation

    # 20200526: 2 options input text or input the all_nouns object, which is a dictionary that relates the key (worform, pos) to the value (noun it transforms into)
    # from scipy.special import comb #scipy is a mathematical packet and comb is the function we need to calculate the entropy
    number = 0 # initiating the sum of concretenesses

    # Convert the text into the noun-wordforms dictionary:
    word_forms = wordformtion(text)
    [nouns, wordforms_notinWordNet] = noun_lemmas(word_forms)

    number_nouns = 0 # initiating the count of nouns in the review

    if repetitions:
        for wordform in word_forms:
            try:
                noun = nouns[wordform] # Note!: noun can be a list of nouns
            except:
                continue
            if noun in exclusion_list: continue
            elif wordform[1][:2] not in postag: continue
            else:
                if type(noun) == list:
                    for n in noun:
                        depth_n = hyp_num(n, wordform[1])
                        if (len(n)<2 or (depth_n == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                        else:
                            frequency = word_forms[wordform] #all_nouns is a dictionary in which the keys are the nouns and their values are their frequencies
                            number = number + depth_n * frequency # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                    number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
                else:
                    try:
                        depth_noun = hyp_num(noun, wordform[1])
                        if wordform[0] in ['gameplay', 'sans', 'microsoft', 'powerpoint', 'youtube', 'hodge', 'pt'] or 'smartophone' in wordform[0]:
                            depth_noun = depth_noun + 1
                        elif wordform[0] in ['ios']:
                            depth_noun = depth_noun + 2
                    except:
                        print("Error in text_depth_lin, unable to calculate noun depth ", "on the noun: ", noun)
                        print("Text of the review: ", text)
                        break
                    if (len(noun) < 2 or (depth_noun == 0 and noun != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                    else:
                        frequency = word_forms[wordform] #all_nouns is a dictionary in which the keys are the nouns and their values are their frequencies
                        number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
                        number = number + depth_noun * frequency # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
    else:
        # Need to create a new nouns subdictionary, which only contains the proper POS tags
        nouns_tag = dict()
        for w, n in nouns.items():
            if w[1][:2] not in postag: continue
            elif n in exclusion_list: continue
            else:
                nouns_tag[w] = n
        # Create a list/set of unique values:
        unique_nouns_tag = []
        for n in nouns_tag.values():
            if n not in unique_nouns_tag:
                unique_nouns_tag.append(n)
        for noun in unique_nouns_tag: # set(nouns_tag.values()):
            # The following commented section calculates the frequency of noun-lemmas (that is it makes equivalent different wordforms which give same noun-lemma - allows for permutations amonfg the distinct wordforms)
            frequency = 1
            # number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review

            if type(noun) == list:
                for n in noun:
                    depth_n = hyp_num(n, 'NN')
                    if (len(n)<2 or (depth_n == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                    else:
                        number = number + depth_n * frequency # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                    number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review
            else:
                depth_noun = hyp_num(noun, 'NN')
                # create a list of wordforms which convert into a given noun:
                wordf_n = [w[0] for w, n in nouns.items() if  n == noun]
                if len(set(wordf_n) & set(['gameplay', 'sans', 'microsoft', 'powerpoint', 'youtube', 'hodge', 'pt', 'smartphone', 'smartphones'])) > 0:
                    depth_noun = depth_noun + 1
                elif len(set(wordf_n) & set(['ios'])) > 0:
                    depth_noun = depth_noun + 2
                if (len(noun)<2 or (depth_noun == 0 and n != 'entity')): continue # make sure the words which are not in WordNet and/or give depth 0 are not included. usually those are incorrectly assigned parts of speech. there was 298 of them without this line of code
                else:
                    number = number + depth_noun * frequency # Note depth_noun needs to be bigger by 1, but 1 is being subtracted
                    number_nouns = number_nouns + frequency #number_nouns is the number of nouns in the review

    return number, number_nouns

def wordforms_to_filter(condition): # 0 - fake, 1 - real, 3 - recall
    # Need to open the .txt file first
    filepath = "common_text.txt"
    handle = open(filepath)
    # The following line of code creates a string object from the text inside the file
    instructions_common = handle.read()

    # Now let's add the instruction file specific to real/fake condition
    if condition != 3:
        if condition == 0:
            file_cond = "fake_text.txt"
        else: file_cond = "real_text.txt"

        handle_cond = open(file_cond)
        instructions_cond = handle_cond.read()

        instructions = instructions_common + instructions_cond
    else:
        instructions = instructions_common

    # print(instructions)

    wordforms_instructions = wordformtion(instructions)
    wordforms_list = [wordform[0] for wordform in wordforms_instructions]
    # print(type(nouns_instructions))
    # print(nouns_instructions)
    return wordforms_list

def nouns_to_filter(condition): # 0 - fa ke, 1 - real, 3 - recall
    # Need to open the .txt file first
    filepath = "common_text.txt"
    handle = open(filepath)
    # The following line of code creates a string object from the text inside the file
    instructions_common = handle.read()

    # Now let's add the instruction file specific to real/fake condition
    if condition != 3:
        if condition == 0:
            file_cond = "fake_text.txt"
        else: file_cond = "real_text.txt"

        handle_cond = open(file_cond)
        instructions_cond = handle_cond.read()

        instructions = instructions_common + instructions_cond
    else:
        instructions = instructions_common

    # print(instructions)

    wordforms_instructions = wordformtion(instructions)
    nouns_instructions = noun_lemmas(wordforms_instructions)[0]
    nouns_list = []
    for n in nouns_instructions.values():
        if n not in nouns_list:
            nouns_list.append(n)
    # print(type(nouns_instructions))
    # print(nouns_instructions)
    return nouns_list

def process_row(row):
    """
    this function receives a single row of a table
    and returns a pair (id, depth) for a given row
    """

    id = row[0]
    reviewtext = row[1]
    if reviewtext == None:
        return (id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    else:
        word_forms = wordformtion(reviewtext)
        word_count = 0
        for wordform in word_forms:
            if wordform[1][0].isalpha():
                word_count += word_forms[wordform]
            else: continue

        [review_depth_N_rep, N_rep] = text_depth(reviewtext, ['NN', 'CD'], True, [])
        [review_depth_N_norep, N_norep] = text_depth(reviewtext, ['NN', 'CD'], False, [])
        [review_depth_V_rep, V_rep] = text_depth(reviewtext, 'VB', True, [])
        [review_depth_V_norep, V_norep] = text_depth(reviewtext, 'VB', False, [])
        [review_depth_J_rep, J_rep] = text_depth(reviewtext, 'JJ', True, [])
        [review_depth_J_norep, J_norep] = text_depth(reviewtext, 'JJ', False, [])
        [review_depth_R_rep, R_rep] = text_depth(reviewtext, 'RB', True, [])
        [review_depth_R_norep, R_norep] = text_depth(reviewtext, 'RB', False, [])

        return  (id, word_count, review_depth_N_rep, N_rep, review_depth_N_norep, N_norep, review_depth_V_rep, V_rep, review_depth_V_norep, V_norep, review_depth_J_rep, J_rep, review_depth_J_norep, J_norep, review_depth_R_rep, R_rep, review_depth_R_norep, R_norep)

def record_answers(currr, answers):
    """
    this function receives cursor to sql (cur) and list of answers List[(id, depth)]
    and records answers to the sql
    for now, this is single process code
    """

    # chunk_size = 50000  # Adjust this value based on your system's performance
    # chunks = [answers[i:i + chunk_size] for i in range(0, len(answers), chunk_size)]
    # print('The number of chunks is: ', len(chunks))
    # count = 1
    # for chunk in chunks:
    #     print('Beginning chunk ', count)
    #     cur.execute('BEGIN TRANSACTION')
    #     for answer in chunk:
    #         id, word_count, review_depth_N_rep, N_rep, review_depth_N_norep, N_norep, review_depth_V_rep, V_rep, review_depth_V_norep, V_norep, review_depth_J_rep, J_rep, review_depth_J_norep, J_norep, review_depth_R_rep, R_rep, review_depth_R_norep, R_norep = answer
    #         cur.execute(''' UPDATE Comments SET Word_count = ?, Depth_N_rep = ?, N_rep = ?, Depth_N_norep = ?, N_norep = ?,
    #     Depth_V_rep = ?, V_rep = ?, Depth_V_norep = ?, V_norep = ?,
    #     Depth_J_rep = ?, J_rep = ?, Depth_J_norep = ?, J_norep = ?,
    #     Depth_R_rep = ?, R_rep = ?, Depth_R_norep = ?, R_norep = ?
    #     WHERE id = ?
    #     ''', (word_count, review_depth_N_rep, N_rep, review_depth_N_norep, N_norep, review_depth_V_rep, V_rep, review_depth_V_norep, V_norep, review_depth_J_rep, J_rep, review_depth_J_norep, J_norep, review_depth_R_rep, R_rep, review_depth_R_norep, R_norep, id))
    #     cur.execute('COMMIT')
    #     print('Ended chunk ', count)
    #     count += 1

    for answer in answers:
        id, word_count, review_depth_N_rep, N_rep, review_depth_N_norep, N_norep, review_depth_V_rep, V_rep, review_depth_V_norep, V_norep, review_depth_J_rep, J_rep, review_depth_J_norep, J_norep, review_depth_R_rep, R_rep, review_depth_R_norep, R_norep = answer
        currr.execute(''' UPDATE Comments SET Word_count = ?, Depth_N_rep = ?, N_rep = ?, Depth_N_norep = ?, N_norep = ?,
        Depth_V_rep = ?, V_rep = ?, Depth_V_norep = ?, V_norep = ?,
        Depth_J_rep = ?, J_rep = ?, Depth_J_norep = ?, J_norep = ?,
        Depth_R_rep = ?, R_rep = ?, Depth_R_norep = ?, R_norep = ?
        WHERE id = ?
        ''', (word_count, review_depth_N_rep, N_rep, review_depth_N_norep, N_norep, review_depth_V_rep, V_rep, review_depth_V_norep, V_norep, review_depth_J_rep, J_rep, review_depth_J_norep, J_norep, review_depth_R_rep, R_rep, review_depth_R_norep, R_norep, id))

    # for answer in answers:
    #     id, word_count, review_depth_N_rep, N_rep, review_depth_N_norep, N_norep, review_depth_V_rep, V_rep, review_depth_V_norep, V_norep, review_depth_J_rep, J_rep, review_depth_J_norep, J_norep, review_depth_R_rep, R_rep, review_depth_R_norep, R_norep = answer
    #     cur_w.execute('UPDATE Comments SET Word_count = ? WHERE id = ?', (word_count, id, ))

    #     cur_w.execute('UPDATE Comments SET Depth_N_rep = ? WHERE id = ?', (review_depth_N_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET N_rep = ? WHERE id = ?', (N_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET Depth_N_norep = ? WHERE id = ?', (review_depth_N_norep,id, ))
    #     cur_w.execute('UPDATE Comments SET N_norep = ? WHERE id = ?', (N_norep,id, ))

    #     cur_w.execute('UPDATE Comments SET Depth_V_rep = ? WHERE id = ?', (review_depth_V_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET V_rep = ? WHERE id = ?', (V_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET Depth_V_norep = ? WHERE id = ?', (review_depth_V_norep,id, ))
    #     cur_w.execute('UPDATE Comments SET V_norep = ? WHERE id = ?', (V_norep,id, ))

    #     cur_w.execute('UPDATE Comments SET Depth_J_rep = ? WHERE id = ?', (review_depth_J_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET J_rep = ? WHERE id = ?', (J_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET Depth_J_norep = ? WHERE id = ?', (review_depth_J_norep,id, ))
    #     cur_w.execute('UPDATE Comments SET J_norep = ? WHERE id = ?', (J_norep,id, ))

    #     cur_w.execute('UPDATE Comments SET Depth_R_rep = ? WHERE id = ?', (review_depth_R_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET R_rep = ? WHERE id = ?', (R_rep,id, ))
    #     cur_w.execute('UPDATE Comments SET Depth_R_norep = ? WHERE id = ?', (review_depth_R_norep,id, ))
    #     cur_w.execute('UPDATE Comments SET R_norep = ? WHERE id = ?', (R_norep,id, ))

if __name__ == '__main__':

    # Main code block
    import sqlite3
    import shutil # we use this library to create a copy of a file (in this case to duplicate the database
    # so that we can loop over one instance while editing the other)
    # Establish a SQLite connection to a database:
    conn = sqlite3.connect('wf_nouns.sqlite') # The copy of the original database to use for iterating
    # # Get the cursor, which is used to traverse the database, line by line
    cur = conn.cursor()
    # Then we duplicate thedatabase, so that one can loop and edit it at the same time
    # and 'open' the other 'instance' of the same database
    shutil.copyfile('wf_nouns.sqlite', 'wf_w.sqlite')
    conn_w = sqlite3.connect('wf_w.sqlite') # The database to be updated
    cur_w = conn_w.cursor()

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Word_count INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Word_count' exists already''')
        pass # handle the error

    # Next we create an extra column in the Comments table to have it edited later with depth.
    # We do that only if the column does not exist yet
    # We create 4 columns for each POS:
    # 1. Column with depth calculated including repetitions
    # 2. Column with the number of terms in the calculation POS_rep or POS_norep
    # 3. 4. the same without repetions
    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_N_rep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_N_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD N_rep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'N_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_N_norep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_N_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD N_norep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'N_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_V_rep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_V_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD V_rep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'V_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_V_norep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_V_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD V_norep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'V_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_J_rep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_J_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD J_rep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'J_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_J_norep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_J_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD J_norep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'J_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_R_rep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_R_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD R_rep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'R_rep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD Depth_R_norep REAL DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'Depth_R_norep' exists already''')
        pass # handle the error

    try:
        cur_w.execute('''ALTER TABLE Comments ADD R_norep INTEGER DEFAULT 0''') # DEFAULT 0 was removed from the sql string
    except:
        print('''The column 'R_norep' exists already''')
        pass # handle the error

    set_wordnet()

    sqlstr = 'SELECT id, Comment_cleaned FROM Comments' # Select query that instructs over what we will be iterating

    rows = [row for row in cur.execute(sqlstr)]   # read rows from sql

    print("start computing..")
    t0 = time.time()

    # n_processes = 28
    n_processes = multiprocessing.cpu_count()

    if n_processes == 1:
        print("single process")
        answers = [process_row(row) for row in rows]  # single process each row in rows
    else:
        print(f"pool process with {n_processes} threads")
        # we call initializer function = set_wordnet so that each worker receives separate wn object
        with Pool(processes=n_processes, initializer=set_wordnet) as pool:
            answers = list(tqdm(pool.imap(process_row, rows), total = len(rows)))

    print(f"finished computing in {time.time() - t0} seconds...")
    # print(len(answers))

    t0 = time.time()
    print("start recording...")
    record_answers(cur_w, answers)   # recording answers
    print(f"finished recording in {time.time() - t0} seconds")

    conn_w.commit()

    cur_w.close()
    conn_w.close()
    cur.close()
    conn.close()

    shutil.copyfile('wf_w.sqlite', 'wf_hypnum.sqlite')
    # shutil.copyfile('wf_hypnum.sqlite', r"C:\Users\i_gordeliy\Dropbox\Marketing\Wayfair\Data\Sqlites\Temp\wf_hypnum.sqlite")