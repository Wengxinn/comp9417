#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 13:33:52 2021

@author: MaggieGuan
"""

import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import re
import textstat
from textstat.textstat import textstatistics, legacy_round

train = pd.read_csv("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/train.csv", header=0)
train = train[['id','excerpt','target','standard_error']]

#add regex to Spacy infix to preserve intra-word concatenators
nlp = spacy.load('en_core_web_sm')
infixes = nlp.Defaults.prefixes + (r'[./]',r"[-]~",r"(.'.)")
infixes_re = spacy.util.compile_infix_regex(infixes)

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, infix_finditer=infixes_re.finditer)

nlp.tokenizer = custom_tokenizer(nlp)

#Define general functions
#break down text into sentences, # of sentences = len(sents list)
def break_sents(text):
    doc = nlp(text)
    return list(doc.sents)

#break down text into tokens, # of words = len(token list)
def break_words(text):
    doc = nlp(text)
    tokens = [token for token in doc if not (token.is_punct or token.is_space)]
    return tokens


#Compute sentence-level features
words_list = [break_words(x) for x in train['excerpt']]
sents_list = [break_sents(x) for x in train['excerpt']]

##count of words(words)
train['words'] = [len(x) for x in words_list]

##characters_per_word(cpw)
train['cpw'] = [np.mean([len(x.text) for x in y]) for y in words_list]

##long_words > 6 characters(lwords)
train['lwords'] = [sum([len(x.text) > 6 for x in y]) for y in words_list]

##syllabi_per_word(spw)
train['spw'] = [np.mean([textstatistics().syllable_count(x.text) for x in l]) for l in words_list]

##polysyllablic > 3 syllables(polysyllab)
train['polysllab'] = [sum([textstatistics().syllable_count(x.text) >= 3 for x in l]) for l in words_list]

##count of sents(sents)
train['sents'] = [len(x) for x in sents_list]

##words per sent(wps)
train['wps'] = train['words']/train['sents']

#Compute document-level features
##Flesch Reading Ease(flesch)
train['flesch'] = [textstat.flesch_reading_ease(x) for x in train['excerpt']]
##Flesch-Kincaid Grade(fleschk)
train['fleschk'] = [textstat.flesch_kincaid_grade(x) for x in train['excerpt']]
##Guning Fog Scale(fog)
train['fog'] = [textstat.gunning_fog(x) for x in train['excerpt']]
##SMOG index(smog)
train['smog'] = [textstat.smog_index(x) for x in train['excerpt']]
##Automated Readability Index(auto)
train['auto'] = [textstat.automated_readability_index(x) for x in train['excerpt']]
##Coleman-Liau Index(cl)
train['cl'] = [textstat.coleman_liau_index(x) for x in train['excerpt']]
##Linear Write Formula(lw)
train['lw'] = [textstat.linsear_write_formula(x) for x in train['excerpt']]
##Dale-Chall Readability Score(dc)
train['dc'] = [textstat.dale_chall_readability_score(x) for x in train['excerpt']]
##standard Readability score based on above tests(standard)
train['standard'] = [textstat.text_standard(x) for x in train['excerpt']]
##LIX: # of words longer than 6 characters / # of words(lix)
train['lix'] = train['lwords']/train['words']
##RIX: # of words longer than 6 characters / # of sents(rix)
train['rix'] = train['lwords']/train['sents']
##Lexical diversity: a measure of how many different words in text(lexdiv)
train['lexdiv'] = [len(set([x.text for x in y])) for y in words_list]/train['words']
##Content diversity: a measure of how many content words in text - adjectives, verb, nouns, adverbs(contdiv) 
train['contdiv'] = [sum([x.pos_ in ['ADJ','ADV','NOUN','VERB'] for x in l]) for l in words_list] / train['words']
##Count of adverb(adv)
train['adv'] = [sum([x.pos_ == 'ADV' for x in l]) for l in words_list] / train['words']
##Count of adjectives(adj)
train['adj'] = [sum([x.pos_ == 'ADJ' for x in l]) for l in words_list] / train['words']
##Count of verb(verb)
train['verb'] = [sum([x.pos_ == 'VERB' for x in l]) for l in words_list] / train['words']
##Count of noun(noun)
train['noun'] = [sum([x.pos_ == 'NOUN' for x in l]) for l in words_list] / train['words']
##Count of pronoun(pron)
train['pron'] = [sum([x.pos_ == 'PRON' for x in l]) for l in words_list] / train['words']
##Incidence of Connectives
##Logic operator connectivity

train.to_csv("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/train_featured.csv", index=False)
