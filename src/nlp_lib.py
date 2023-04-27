import spacy
import neuralcoref

from src.constants import Tags

""" 
Custom spaCy language parser with NeuralCoref
"""
def custom_sentencizer(doc):
    """
    # use customized sentence start and ends
    # see: https://stackoverflow.com/questions/57660268/how-to-specify-spacy-to-recognize-a-sentence-based-on-full-stop
    """
    for i, token in enumerate(doc[:-2]):  # last token can not start a sentence
        if token.text == "\n":
            doc[i + 1].is_sent_start = True
        else:
            doc[i + 1].is_sent_start = False  # default sentencizer should ignore this token

    return doc


def get_parser():
    nlp = spacy.load("en_core_web_lg")

    # add custom sentencizer to pipeline in front of the parser itself
    nlp.add_pipe(custom_sentencizer, before="parser")

    # add neuralcoref to pipeline to enable pronoun resolution
    neuralcoref.add_to_pipe(nlp)

    return nlp


print("building the language parser ...")
nlp = get_parser()



"""
Custom method of forwarding input (allow processing of list of sentences) to the language parser 
"""
def parse(sentences: list):
    """
    use language parser to parse sentences
    """
    n_sentences = len(sentences)
    text = concat_sentences(sentences, n_sentences)
    doc = nlp(text)

    # check whether the number of sentences from doc is equal to the expected number of sentences
    assert sum(1 for _ in enumerate(doc.sents)) == n_sentences, \
        "expected {n_sentences} sentences, but spaCy found {sum(1 for _ in enumerate(doc.sents))}:\n{text}"

    return doc


def concat_sentences(sentences: list, n_sentences: int):
    """
    concat multiple sentences to a longer text
    """
    sentences = [process_sentence(s) for s in sentences]
    if len(sentences) != n_sentences:
        exit(f"{list} does not contain exactly {n_sentences}.")
    text = ""
    for i in range(n_sentences):
        text = text + process_sentence(sentences[i])

    return text


def process_sentence(sentence: str):
    """
    process a sentence (remove white spaces)
    """
    sentence = sentence.replace("<unk>", "token_unknown")  # For MT model output sentences
    # the following lines are there to enable the custom_sentencizer work correctly
    # (new lines show the end of sentences, not puncts)
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace(".", " ")
    sentence = sentence.strip() + "\n"

    return sentence


"""
Further functionality provided by spaCy
"""
def pronoun_resolution(token: spacy.tokens.Token, doc: spacy.tokens.Doc):
    """
    perform pronoun resolution (using determined clusters by NeuralCoref) for a given pronoun token
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
