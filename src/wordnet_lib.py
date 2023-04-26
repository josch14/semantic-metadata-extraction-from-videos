import nltk
nltk.download('wordnet', quiet=True)

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


NOUN, VERB, ADJ, ADV = "noun", "verb", "adj", "adv"


class WordNetDictionary:
    """
    class for building the WordNet vocab before using it
    """

    def __init__(self):
        print("building WordNet vocab ...")
        self.nouns = self.__get_words_of_type(wn.NOUN)
        self.verbs = self.__get_words_of_type(wn.VERB)
        self.adjectives = self.__get_words_of_type(wn.ADJ)
        self.adverbs = self.__get_words_of_type(wn.ADV)
        self.lemmatizer = WordNetLemmatizerWrapped()


    @staticmethod
    def __get_words_of_type(word_type: str):
        """
        get all WordNet words of a specific type (wn.NOUN, wn.VERB, wn.ADJ, or wn.ADV)
        """
        word_set = set()
        for word in set(wn.words()):
            if len(wn.synsets(word, word_type)) > 0:
                word_set.add(word)

        return word_set


    def is_wordnet_noun(self, token):
        noun_str = token.text.lower()
        return noun_str in self.nouns or self.lemmatizer.lemmatize_noun(noun_str) in self.nouns

    def is_wordnet_verb(self, token):
        verb_str = token.text.lower()
        return verb_str in self.verbs or self.lemmatizer.lemmatize_verb(verb_str) in self.verbs

    def is_wordnet_adjective(self, token):
        adj_str = token.text.lower()
        return adj_str in self.adjectives or self.lemmatizer.lemmatize_adjective(adj_str) in self.adjectives

    def is_wordnet_adverb(self, token):
        adv_str = token.text.lower()
        return adv_str in self.adverbs or self.lemmatizer.lemmatize_adverb(adv_str) in self.adverbs



class WordNetLemmatizerWrapped:
    """
    wrapped WordNet lemmatizer in order to add custom lemmatizations
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()


    def lemmatize_noun(self, noun: str):
        noun = noun.lower()

        # custom lemmatization
        # e.g.: "men" should be lemmatized to "man", which is not performed by WordNet
        if noun == "men":
            return "man"
        if noun == "bikers":
            return "biker"
        if noun == "businessmen":
            return "businessman"

        wn_lemma = self.lemmatizer.lemmatize(noun, wn.NOUN)

        return wn_lemma


    def lemmatize_verb(self, verb: str):
        verb = verb.lower()

        if verb in ["riding", "rides", "rode", "ride"]:
            return "ride"
        if verb in ["staring", "stares", "stared", "stare"]:
            return "stare"
        if verb in ["taping", "tapes", "taped", "tape"]:
            return "tape"
        if verb in ["falling", "falls", "fell", "fall"]:
            return "fall"
        if verb in ["bathing", "bathes", "bathed", "bathe"]:
            return "bathe"
        if verb in ["scraping", "scrapes", "scraped", "scrape"]:
            return "scrape"
        if verb in ["shining", "shines", "shone", "shine"]:
            return "shine"
        if verb in ["seeing", "sees", "saw", "see"]:
            return "see"
        if verb in ["feeding", "feeds", "fed", "feed"]:
            return "feed"
        if verb in ["moping", "mopes", "moped", "mope"]:
            return "mope"
        if verb in ["plating", "plates", "plated", "plate"]:
            return "plate"
        if verb in ["rating", "rates", "rated", "rate"]:
            return "rate"

        wn_lemma = self.lemmatizer.lemmatize(verb, wn.VERB)

        return wn_lemma


    def lemmatize_adjective(self, adj: str):
        adj = adj.lower()

        # no custom lemmatizations
        wn_lemma = self.lemmatizer.lemmatize(adj, wn.ADJ)

        return wn_lemma


    def lemmatize_adverb(self, adv: str):
        adv = adv.lower()

        # no custom lemmatizations
        wn_lemma = self.lemmatizer.lemmatize(adv, wn.ADV)

        return wn_lemma



def get_words_from_synsets(word: str, word_type: str, verbose: bool = False):
    """
    get all WordNet words from the given word's synsets
    """
    if word_type == NOUN:
        word_type = wn.NOUN
    elif word_type == VERB:
        word_type = wn.VERB
    elif word_type == ADJ:
        word_type = wn.ADJ
    elif word_type == ADV:
        word_type = wn.ADV
    else:
        exit("word type not known by WordNet")


    if verbose:
        print(word)

    lemmas = []
    for synset in wn.synsets(word, pos=word_type):
        if verbose:
            print(synset)
            print(synset.lemma_names())

        lemmas = lemmas + synset.lemma_names()

    lemmas = list(set(lemmas))
    lemmas = [l.lower() for l in lemmas]

    if verbose:
        print(lemmas)

    return lemmas
