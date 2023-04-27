import spacy

from .constants import Tags, Dependencies
from .entities_lib import EntitiesLib
from .nlp_lib import pronoun_resolution
from .semantic_metadata.event_relation import EventRelation
from .semantic_metadata.video_relation import VideoRelation
from .wordnet_lib import WordNetDictionary, WordNetLemmatizerWrapped


VERB_TAGS = [Tags.VERB, Tags.AUX]
PASSIVE_VERBS_DEPS = [Dependencies.NSUBJPASS, Dependencies.AUXPASS, Dependencies.AGENT]

VERB_MODIFIERS = [
    (Dependencies.PRT, Tags.ADP),
    (Dependencies.PREP, Tags.ADP)]

VERB_MODIFIERS_FOR_POBJ = VERB_MODIFIERS + [
    (Dependencies.ADVMOD, Tags.ADV),
    (Dependencies.CONJ, Tags.ADV),
    (Dependencies.CONJ, Tags.ADP)]

PREPOSITIONS = [
    "in", "into", "around", "on", "with", "to",
    "down", "onto", "for", "up", "out", "by",
    "behind", "across", "off", "of", "at", "toward",
    "along", "from", "over", "inside", "under",
    "through", "during", "about", "against",
    "between", "after", "towards", "beside", "above",
    "past", "outside", "before", "throughout",
    "alongside", "without", "next", "atop", "within",
    "among", "amongst", "underneath", "until", "due",
    "via", "beneath", "back", "beyond", "unto",
    "amidst", "below", "opposite"]


class RelationsLib:

    @staticmethod
    def extract_relations(doc: spacy.tokens.Doc,
                          timestamps: list,
                          wn_dictionary: WordNetDictionary,
                          wn_lemmatizer: WordNetLemmatizerWrapped):
        """
        core functionality of this class:
        extract event-level and video-level relations from a spaCy doc and list of timestamps
        """
        # determine entities
        noun_compounds, tokens_for_noun_compounds, nouns, tokens_for_nouns, tokens_for_entities = \
            EntitiesLib.extract_nouns_from_doc(doc, wn_dictionary, wn_lemmatizer)
        entity_names = noun_compounds + nouns
        entity_tokens = tokens_for_noun_compounds + tokens_for_nouns

        # inform the user when the input text does not contain any verb
        if not any([token.pos_ in VERB_TAGS for token in doc]):
            print(f"\nno relation can be extracted (input text does not contain any verb): \n{doc}")
            return [], []

        # 1) search for verbs
        verbs = []
        for token in doc:
            if token.pos_ in VERB_TAGS:

                if list(filter(lambda child: child.dep_ in PASSIVE_VERBS_DEPS, list(token.children))):
                    # do not use passive verbs
                    continue

                # verb is valid only when
                # (a) it is known by WordNet OR
                # (b) verb is something like "'s" and "'re", which is found by spaCy as verb and lemmatized to "be".
                #   Such verbs are not recognized by WordNet as verbs. In this case, we ignore WordNet
                if wn_dictionary.is_wordnet_verb(token) or token.lemma_ == "be":
                    verbs.append(token)

        # 2) search for fitting subject
        tuples = []
        for verb in verbs:
            subject = RelationsLib.__find_subject_for_verb(verb, tokens_for_entities)
            if subject is not None:
                tuples.append((subject, verb))

        # 3) search for fitting objects
        candidate_relations = []
        for subject, verb in tuples:
            modifiers_of_objects, objects = RelationsLib.__find_objects_for_verb(verb, tokens_for_entities, wn_dictionary)
            for modifiers, object in zip(modifiers_of_objects, objects):
                candidate_relations.append((subject, verb, modifiers, object))

        # process the potential relations
        # 1) for both subject and object look for conjunctions, each resulting in a list of tokens (usually consisting of a single token)
        relations = []
        for subject, verb, modifiers, object in candidate_relations:
            relations.append([
                RelationsLib.__find_conjunct_tokens_of_entities_or_prons(subject, tokens_for_entities),
                verb,
                modifiers,
                RelationsLib.__find_conjunct_tokens_of_entities_or_prons(object, tokens_for_entities)
            ])

        # 2) finalize the relations
        video_level_relations = []
        event_level_relations = []
        for subjects, verb, modifiers, objects in relations:

            # for each relation, get the index of the sentence to which it belongs (using the corresponding verb)
            # this is done to link the relations to the corresponding timestamp later
            sentence_index = RelationsLib.__get_sentence_idx_for_token(doc, verb)
            assert all([sentence_index == RelationsLib.__get_sentence_idx_for_token(doc, t) for t in subjects]) \
                   and all([sentence_index == RelationsLib.__get_sentence_idx_for_token(doc, t) for t in modifiers]) \
                   and all([sentence_index == RelationsLib.__get_sentence_idx_for_token(doc, t) for t in objects]), \
                "all tokens of a relation should occur in the same sentence"

            # apply pronoun resolution
            subjects = [pronoun_resolution(t, doc) for t in subjects]
            objects = [pronoun_resolution(t, doc) for t in objects]

            # pronoun resolution will fail in cases, i.e., relations may still contain pronouns
            # when a pronoun can not be resolved, it is removed
            subjects = [t for t in subjects if t in tokens_for_entities]
            objects = [t for t in objects if t in tokens_for_entities]
            if len(subjects) == 0 or len(objects) == 0:
                continue

            # produce string representations
            subjects_string_list = RelationsLib.__convert_entity_tokens_to_string_list(subjects, entity_names, entity_tokens)
            verb_string = verb.text if wn_dictionary.is_wordnet_verb(verb) else verb.lemma_
            modifiers_string_list = [t.text for t in modifiers]
            objects_string_list = RelationsLib.__convert_entity_tokens_to_string_list(objects, entity_names, entity_tokens)
            timestamp = timestamps[sentence_index]

            video_level_relation = VideoRelation(
                subjects=subjects_string_list,
                verb=verb_string,
                modifiers=modifiers_string_list,
                objects=objects_string_list
            )
            if video_level_relation not in video_level_relations:
                video_level_relations.append(video_level_relation)

            event_level_relation = EventRelation(
                subjects=subjects_string_list,
                verb=verb_string,
                modifiers=modifiers_string_list,
                objects=objects_string_list,
                timestamp=timestamp)
            if event_level_relation not in event_level_relations:
                event_level_relations.append(event_level_relation)

        return video_level_relations, event_level_relations


    """
    all sorts of helper function to provide the core functionality of entity and property extraction
    """
    @staticmethod
    def __is_preposition(token: spacy.tokens.Token):
        return token.text.lower() in PREPOSITIONS


    @staticmethod
    def __is_entity_or_pronoun(token: spacy.tokens.Token, tokens_for_entities: list):
        return token in tokens_for_entities or token.pos_ == Tags.PRON


    @staticmethod
    def __find_subject_for_verb(verb: spacy.tokens.Token, tokens_for_entities: list):
        """
        for the input verb, find the corresponding subjects
        """
        # 1) the verb itself has a child with a desired subject dependency
        # (prioritize entities before pronouns, only necessary if spaCy does weird things)

        # 1.1) noun is an entity
        for child in verb.children:
            if child.dep_ == Dependencies.NSUBJ and child in tokens_for_entities:
                return child
        # 1.2) noun is a pronoun
        for child in verb.children:
            if child.dep_ == Dependencies.NSUBJ and child.pos_ == Tags.PRON:
                return child


        # 2) if the verb itself has the dependency "acl" and parent NOUN, then the parent is the subject
        if verb.dep_ == Dependencies.ACL and RelationsLib.__is_entity_or_pronoun(verb.head, tokens_for_entities):
            return verb.head

        # 3) search recursively for a parent that has a desired subject dependency
        subject = RelationsLib.__find_subject_of_parent(verb.head, tokens_for_entities)
        if subject is not None:
            return subject

        return None


    @staticmethod
    def __find_subject_of_parent(parent: spacy.tokens.Token, tokens_for_entities: list):
        """
        search recursively for a token that has a desired subject dependency
        """
        SUBJECT_DEPS = [Dependencies.NSUBJ, Dependencies.NSUBJPASS]

        def is_subject(token: spacy.tokens.Token, tokens_for_entities: list):
            # either the token is an entity or it's a pronoun, on which we may be allowed to use pronoun resolution later on
            return token.dep_ in SUBJECT_DEPS and RelationsLib.__is_entity_or_pronoun(token, tokens_for_entities)

        # parent may be a subject
        if is_subject(parent, tokens_for_entities):
            return parent

        # subject may be any child of the parent (prioritize entities before pronouns)
        # for example, nsubj are children of verbs
        subjects = list(filter(lambda child: is_subject(child, tokens_for_entities), list(parent.children)))
        if len([s for s in subjects if s in tokens_for_entities]) > 0:
            return [s for s in subjects if s in tokens_for_entities][0]
        if len(subjects) > 0:
            return subjects[0]

        # no subject found on this height
        # if possible, search for subjects using the new parent token
        if parent == parent.head:
            return None
        else:
            return RelationsLib.__find_subject_of_parent(parent.head, tokens_for_entities)


    @staticmethod
    def __find_objects_for_verb(verb: spacy.tokens.Token, tokens_for_entities: list, wn_dictionary: WordNetDictionary):
        """
        for the input verb, find the corresponding objects
        """
        objects, modifiers_of_objects = [], []

        # 1) dobj: direct objects
        for child in verb.children:
            if child.dep_ == Dependencies.DOBJ and RelationsLib.__is_entity_or_pronoun(child, tokens_for_entities):
                objects.append(child)
                modifiers_of_objects.append([])

        # 2) pobj: objects of preposition
        pobjs, modifiers_of_pobjs = RelationsLib.__find_pobj(verb, verb, tokens_for_entities, wn_dictionary)
        # when a coordinating conjunction (Dependencies.CONJ) was used for finding a pobj, then we split the resulting
        # relation up into two relations (and remove the coordinating conjunction from the modifiers list)

        new_pobjs, modifiers_of_new_pobjs = [], []
        for pobj, modifiers_of_pobj in zip(pobjs, modifiers_of_pobjs):

            conj_token = [t for t in modifiers_of_pobj if t.dep_ == Dependencies.CONJ]

            # second case necessary as sometimes, although there is a CONJ, it's the only modifier
            # maybe sometimes happens because of a confusion of SCONJ with ADP
            if len(conj_token) == 0 or len(modifiers_of_pobj) == 1:
                continue

            conj_token = conj_token[0]
            parent_token = conj_token.head

            # nothing to split up
            if parent_token not in modifiers_of_pobj:
                continue

            # new relation
            new_modifiers_of_pobj = [t for t in modifiers_of_pobj if t != parent_token]
            modifiers_of_new_pobjs.append(new_modifiers_of_pobj)
            new_pobjs.append(pobj)

            # remove the CONJ from the other modifier list of other relation
            modifiers_of_pobj.remove(conj_token)

            assert len(modifiers_of_pobj) == len(new_modifiers_of_pobj), \
                "when using conj in pobj search, then the corresponding tokens on paths should have the same length"
        pobjs += new_pobjs
        modifiers_of_pobjs += modifiers_of_new_pobjs

        # finish the pobj finding process
        for pobj, modifiers_of_pobj in zip(pobjs, modifiers_of_pobjs):
            objects.append(pobj)
            modifiers_of_objects.append(modifiers_of_pobj)

        # now for both cases (dobj and pobj) search for verb particles (often prt, sometimes prep) and add those
        # to the additional token list. We assume such to be leafs of the verb (child tokens without any further children)
        leaf_modifiers = RelationsLib.__find_leaf_modifiers(verb)
        leaf_modifiers = [t for t in leaf_modifiers if t.pos_ == Tags.ADP and RelationsLib.__is_preposition(t)]
        for modifiers_of_object in modifiers_of_objects:
            modifiers_of_object += leaf_modifiers
            RelationsLib.__sort_token_list(modifiers_of_object)

        return modifiers_of_objects, objects


    @staticmethod
    def __sort_token_list(tokens):
        tokens.sort(key=lambda tok: tok.i, reverse=False)


    @staticmethod
    def __find_pobj(token: spacy.tokens.Token,
                  root_verb: spacy.tokens.Token,
                  tokens_for_entities: list,
                  wn_dictionary: WordNetDictionary):
        """
        find objects of preposition
        """
        pobjs, modifier_lists = [], []

        for child in token.children:
            if root_verb != token and child.dep_ == Dependencies.POBJ \
                    and RelationsLib.__is_entity_or_pronoun(child, tokens_for_entities):
                # at least one token has to be between verb and pobj (root_verb != token)
                # pobj found, i.e., return it and the current token, do not search any deeper
                return [child], [[token]]

            # no pobj found, i.e., search recursively by using method on all child tokens
            if (child.dep_, child.pos_) in VERB_MODIFIERS_FOR_POBJ:
                # check whether we have a valid adverb or preposition
                if child.pos_ == Tags.ADV and not wn_dictionary.is_wordnet_adverb(child):
                    continue
                elif child.pos_ == Tags.ADP and not RelationsLib.__is_preposition(child):
                    print(f"preposition {child} not known. Add it to PREPOSITIONS in entities_lib.py if desired.")
                    continue
                pobjs_rec, modifiers_rec = RelationsLib.__find_pobj(child, root_verb, tokens_for_entities, wn_dictionary)
                pobjs += pobjs_rec
                modifier_lists += modifiers_rec

        # if there is a pobj (modifiers will have at least one element), add the current token to the modifier list
        if token != root_verb:
            for modifiers in modifier_lists:
                modifiers.append(token)

        # validate
        for modifiers in modifier_lists:
            assert len([t for t in modifiers if (t.dep_, t.pos_) not in VERB_MODIFIERS_FOR_POBJ]) == 0, \
                "error in pobj"

        return pobjs, modifier_lists


    @staticmethod
    def __find_leaf_modifiers(verb: spacy.tokens.Token):
        """
        when at least one object was found for a verb, this method is called to search for leaf modifiers
        """
        modifiers = []
        for child in verb.children:
            if (child.dep_, child.pos_) in VERB_MODIFIERS:
                # child is a verb_modifier when it has no children
                n_modifier_children = 0
                for _ in child.children:
                    n_modifier_children += 1

                if n_modifier_children == 0:
                    modifiers.append(child)

        return modifiers


    @staticmethod
    def __convert_entity_tokens_to_string_list(tokens: list, entity_names: list, entity_tokens: list):
        assert len(tokens) != 0, "length of entity tokens should never be 0"

        entity_strings = []
        for t in tokens:
            for name, tokens_for_name in zip(entity_names, entity_tokens):
                if t in tokens_for_name:
                    entity_strings.append(name)
                    break

        assert len(tokens) == len(entity_strings)

        return entity_strings


    @staticmethod
    def __find_conjunct_tokens_of_entities_or_prons(token: spacy.tokens.Token, tokens_for_entities: list):
        conjunct_tokens = [token]
        for child in token.children:
            if child.dep_ == Dependencies.CONJ and RelationsLib.__is_entity_or_pronoun(child, tokens_for_entities):
                conjunct_tokens += RelationsLib.__find_conjunct_tokens_of_entities_or_prons(child, tokens_for_entities)
        return conjunct_tokens


    @staticmethod
    def __get_sentence_idx_for_token(doc: spacy.tokens.Doc, token: spacy.tokens.Token):
        """
        find the sentence to which a token belongs
        """
        for idx, sent in enumerate(doc.sents):
            if token in sent:
                return idx

        assert 0 == 1, "token {token} not in doc {doc}"
