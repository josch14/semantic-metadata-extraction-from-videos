import spacy.tokens

from src.semantic_metadata.entity_property import EntityPropertyPair
from .constants import Tags, Dependencies
from .semantic_metadata.event_entity import EventEntity
from .semantic_metadata.video_entity import VideoEntity

from .wordnet_lib import WordNetLemmatizerWrapped, WordNetDictionary


NOUN_TAGS = [Tags.NOUN, Tags.PROPN]


def extract_entities_and_properties(doc: spacy.tokens.Doc,
                                    timestamps: list,
                                    wn_dictionary: WordNetDictionary,
                                    wn_lemmatizer: WordNetLemmatizerWrapped):
    """
    extract event-level and video-level entities from a spaCy doc and a list of timestamps.
    """
    # get entities
    noun_compounds, tokens_for_noun_compounds, nouns, tokens_for_nouns, tokens_for_entities = extract_nouns_from_doc(
        doc, wn_dictionary, wn_lemmatizer
    )
    entity_names = noun_compounds + nouns
    entity_tokens = tokens_for_noun_compounds + tokens_for_nouns


    video_level_entities = []
    event_level_entities = []
    entity_property_pairs = []

    # 1) process nouns to entities, find properties of entities
    for entity_name, tokens in zip(entity_names, entity_tokens):
        sentence_index = get_sentence_idx_of_token(tokens[0], doc)

        # 1.1) entities
        video_entity = VideoEntity(entity_name)
        if video_entity not in video_level_entities:
            video_level_entities.append(video_entity)

        event_entity = EventEntity(entity_name, timestamps[sentence_index])
        if event_entity not in event_level_entities:
            event_level_entities.append(event_entity)

        # 1.2) Properties
        properties = get_properties_for_tokens(tokens, wn_dictionary, wn_lemmatizer)
        for property in properties:
            pair = EntityPropertyPair(video_entity.name, property)  # properties are lemmatized already
            if pair not in entity_property_pairs:
                entity_property_pairs.append(pair)

    # 2) process resolved pronouns for further event-level entities
    # for each pronoun, we need to know the sentence in which occurs to get the correct timestamps later on
    resolved_pronouns = [
        (pronoun_resolution(t, doc), get_sentence_idx_of_token(t, doc)) for t in doc if t.pos_ == Tags.PRON
    ]
    # filter out non-resolved pronouns
    resolved_pronouns = [
        (t, time_idx) for (t, time_idx) in resolved_pronouns if t.pos_ in NOUN_TAGS
    ]

    for (n, time_idx) in resolved_pronouns:
        # get the correct noun from the token (can't use token.text because the noun may be a compound noun)
        noun_from_pronoun = None
        for idx, token_list in enumerate(entity_tokens):
            if n in token_list:
                noun_from_pronoun = entity_names[idx]

        if noun_from_pronoun is None:
            # this case happens when the resolved pronoun, i.e., now a token with noun tag, is not a WordNet noun
            # this may occur due to mispells (e.g., pesron instead of person)
            continue

        # no (resolved) pronoun can provide a new property
        event_entity = EventEntity(noun_from_pronoun, timestamps[time_idx])
        if event_entity not in event_level_entities:
            event_level_entities.append(event_entity)

    video_level_entities.sort(key=sort_by_name)
    event_level_entities.sort(key=sort_by_timestamp)
    entity_property_pairs.sort(key=sort_by_name_of_ep)

    # do not make entities unique because, e.g., "parking_lot" can be in a sentence 2 times
    return video_level_entities, event_level_entities, entity_property_pairs


def sort_by_name(e):
    return e.name


def sort_by_timestamp(e):
    return e.timestamp[0]


def sort_by_name_of_ep(ep):
    return ep.entity


def get_sentence_idx_of_token(token: spacy.tokens.Token, doc: spacy.tokens.Doc):
    for i, sent in enumerate(doc.sents):
        if token in sent:
            return i
    return None


def extract_nouns_from_doc(doc: spacy.tokens.Doc, wn_dictionary: WordNetDictionary, wn_lemmatizer: WordNetLemmatizerWrapped):
    # 1) compound Nouns
    noun_compounds, tokens_for_noun_compounds = get_noun_compounds(doc, wn_dictionary, wn_lemmatizer)
    tokens_for_entities = []
    for token_list in tokens_for_noun_compounds:  # list of lists
        tokens_for_entities += token_list

    # 2) nouns
    nouns, tokens_for_nouns = get_nouns_from_doc_spacy_and_wordnet(doc, wn_dictionary, tokens_for_entities)
    for token_list in tokens_for_nouns:  # list of lists
        tokens_for_entities += token_list

    return noun_compounds, tokens_for_noun_compounds, nouns, tokens_for_nouns, tokens_for_entities


def get_nouns_from_doc_spacy_and_wordnet(doc: spacy.tokens.Doc, wn_dictionary: WordNetDictionary, ignore_tokens: list = None):
    """
    noun detection from a spaCy doc
    """
    nouns = []
    used_tokens = []
    for token in doc:
        if ignore_tokens is not None and token in ignore_tokens:
            # skip this token, probably because it was used for a noun compound
            continue

        if is_spacy_noun(token) and wn_dictionary.is_wordnet_noun(token):
            nouns.append(token.text.lower())  # add the noun non-lemmatized
            used_tokens.append([token])

    return nouns, used_tokens


def is_spacy_noun(token: spacy.tokens.Token):
    """
    check if a word is a spaCy noun.
    """
    return token.pos_ in NOUN_TAGS


def get_noun_compounds(doc: spacy.tokens.Doc, wn_dictionary: WordNetDictionary, wn_lemmatizer: WordNetLemmatizerWrapped):
    """
    compound noun detection from a spaCy doc
    """
    # collect potential compounds
    compound_tokens = get_potential_compounds(doc)

    noun_compounds = []
    used_tokens = []
    for potential_compound in compound_tokens:
        # check potential compounds
        noun_compound_help, used_tokens_help = get_noun_compound_from_potential_compound(
            potential_compound,
            wn_dictionary,
            wn_lemmatizer
        )

        if noun_compound_help is not None:
            noun_compounds.append(noun_compound_help)
            used_tokens.append(used_tokens_help)

    return noun_compounds, used_tokens


def get_potential_compounds(doc: spacy.tokens.Doc):
    """
    helper function for compound noun detection.
    """
    compound_roots = set()
    # collect all heads of potential compounds (not necessarily a noun)
    for token in doc:
        if token.dep_ != Dependencies.COMPOUND:
            comps = [child.text for child in token.children if child.dep_ == Dependencies.COMPOUND]
            if len(comps) > 0:
                compound_roots.add(token)

    # for each head, recursively collect all children with compound dependency (not necessarily nouns)
    compound_tokens = [None] * len(compound_roots)
    for idx, compound_root in enumerate(compound_roots):
        compound_tokens[idx] = [compound_root]
        while True:
            new_sub_compound_tokens = 0
            for compound_token in compound_tokens[idx]:
                for child in compound_token.children:
                    if child.dep_ == Dependencies.COMPOUND and child not in compound_tokens[idx]:
                        new_sub_compound_tokens += 1
                        compound_tokens[idx].append(child)
            if new_sub_compound_tokens == 0:
                break

    # sort tokens for each compound and return tokens
    potential_compounds = []
    for compound in compound_tokens:
        compound.sort(key=lambda t: t.i)
        potential_compounds.append([token for token in compound])

    return potential_compounds


def get_noun_compound_from_potential_compound(compound_tokens: list,
                                              wn_dictionary: WordNetDictionary,
                                              wn_lemmatizer: WordNetLemmatizerWrapped):
    """
    helper function for compound noun detection.
    """

    n_tokens = len(compound_tokens)
    while n_tokens > 1:
        i = 0
        found_compound = ""
        while i + n_tokens <= len(compound_tokens):
            candidate_token_list = compound_tokens[i:i + n_tokens]

            # check if at least one token is a NOUN or a PROPN
            if len([t for t in candidate_token_list if is_spacy_noun(t)]) > 0:
                candidate_str_list = [token.text.lower() for token in candidate_token_list]
                open_compound = '_'.join(candidate_str_list)  # peanut_butter
                hyphenated_compound = '-'.join(candidate_str_list)  # t-shirt
                regular_compound = ''.join(candidate_str_list)  # weightlifting

                wn_nouns = wn_dictionary.nouns
                if open_compound in wn_nouns or wn_lemmatizer.lemmatize_noun(open_compound) in wn_nouns:
                    found_compound = open_compound

                elif hyphenated_compound in wn_nouns or wn_lemmatizer.lemmatize_noun(
                        hyphenated_compound) in wn_nouns:
                    found_compound = hyphenated_compound

                elif regular_compound in wn_nouns or wn_lemmatizer.lemmatize_noun(
                        regular_compound) in wn_nouns:
                    found_compound = regular_compound

            if found_compound != "":
                # Return the noun lower-cased and non lemmatized
                return found_compound.lower(), compound_tokens[i:i + n_tokens]

            i += 1

        # so far, no compound found -> search one length smaller
        n_tokens -= 1

    return None, []




def get_properties_for_tokens(entity_token_list, wn_dictionary: WordNetDictionary, wn_lemmatizer: WordNetLemmatizerWrapped):
    """
    entity-property pair detection from a token list of an entity
    """

    properties = []
    for token in entity_token_list:
        for child in token.children:
            # Don't filter for spaCy ADJ only as there are quite some good amods (that are WordNet adjectives),
            # but spaCy e.g. VERBS or PROPNS (white, blue, black (PROPN), injured, smiling (VERB))
            if child not in entity_token_list and child.dep_ == Dependencies.AMOD and wn_dictionary.is_wordnet_adjective(
                    child):
                # Add the adjective lemmatized
                properties.append(wn_lemmatizer.lemmatize_adjective(child.text))
    return properties


def pronoun_resolution(token: spacy.tokens.Token, doc: spacy.tokens.Doc):
    """
    resolution for pronouns
    """
    if token.pos_ != Tags.PRON:
        return token

    # we have a pronoun -> try pronoun resolution
    if not doc._.has_coref:
        return token

    cluster_of_token = None
    for cluster in doc._.coref_clusters:
        for mention in cluster.mentions:
            if token in mention:
                cluster_of_token = cluster
                break
        if cluster_of_token:
            break

    if not cluster_of_token:
        return token

    return cluster.main.root
