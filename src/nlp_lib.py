import spacy
import neuralcoref
from pathlib import Path

""" 
Custom spaCy language parser with NeuralCoref
"""
# Use customized sentence start and ends. See:
# https://stackoverflow.com/questions/57660268/how-to-specify-spacy-to-recognize-a-sentence-based-on-full-stop


def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):  # The last token cannot start a sentence
        if token.text == "\n":
            doc[i + 1].is_sent_start = True
        else:
            doc[i + 1].is_sent_start = False  # Tell the default sentencizer to ignore this token
    return doc


def get_parser():
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(custom_sentencizer,
                 before="parser")  # Insert sentencizer before the parser can build its own sentences
    neuralcoref.add_to_pipe(nlp)
    return nlp


print("building the language parser ...")
nlp = get_parser()


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



"""
Important methods for processing of list of sentences / text to the parser
"""
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
    # the following lines are there to enable the custom_sentencizer work correctly (new lines show the end
    # of sentences, not puncts)
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace(".", " ")
    sentence = sentence.strip() + "\n"
    return sentence


"""
Further helpful methods- not relevant for any of the extraction methods
"""
def saveDependencyTree(sentence):
    """
    save the dependency tree (created by displacy) to a svg file
    """
    doc = nlp(sentence)
    svg = spacy.displacy.render(doc, style="dep", jupyter=False)
    file_name = '_'.join([w.text for w in doc if not w.is_punct]) + ".svg"
    file_name = file_name.replace("\n", "")

    if len(file_name) > 160:
        file_name = file_name[:150] + ".svg"

    workingDir = Path().resolve()
    output_path = Path(str(workingDir) + "/depTrees/" + file_name)

    Path(str(workingDir) + "/depTrees/").mkdir(parents=True, exist_ok=True)  # Create folder if not existent
    output_path.open("w", encoding="utf-8").write(svg)  # Write file


def printTokenizedSentence(sentence):
    doc = nlp(sentence)

    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)


def printTokensDetailed(tokens):
    if isinstance(tokens, list):
        print("Detailed Print of Tokens: {}".format(tokens))
        for token in tokens:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)
    else:
        print(tokens.text, tokens.lemma_, tokens.pos_, tokens.tag_, tokens.dep_)


def printTreeProperties(sentence):
    print(sentence)
    for token in nlp(sentence):
        print("{} [{}]".format(token, token.pos_))
        print("Children: {}".format(list(token.children)))
        print("Parents: {}".format(list(token.ancestors)))
        print("Lefts: {}".format(list(token.lefts)))
        print("Rights: {}".format(list(token.rights)))
        print()


def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts],
              [t.orth_ for t in tok.rights])
