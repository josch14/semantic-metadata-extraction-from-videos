import argparse

from src import nlp_lib, relations_lib
from src.entities_lib import extract_entities_and_properties
from src.wordnet_lib import WordNetDictionary, WordNetLemmatizerWrapped

parser = argparse.ArgumentParser()
# make sure to use "." to end sentences, and only to end sentences
parser.add_argument('-t', '--text', type=str, required=True)
args = parser.parse_args()


if __name__ == "__main__":
    text = args.text
    if text[-1] != ".":
        exit("Error: please make sure to use \".\" to end sentences, and only to end sentences!")
    text = text[:-1]
    sentences = text.split(". ")

    # The following is necessary to feed them to the language parser 
    timestamps = []
    for i in range(len(sentences)):
        timestamps.append(([0.0, 0.0]))

    # load WordNet
    wn_dictionary = WordNetDictionary()
    wn_lemmatizer = WordNetLemmatizerWrapped()

    # create linguistic annotations using the language parser
    doc = nlp_lib.parse(sentences)


    # 1) extract video- and event-level entities and entity-property pairs
    video_level_entities, _, entity_property_pairs = extract_entities_and_properties(
        doc, timestamps, wn_dictionary, wn_lemmatizer
    )

    # 2) extract video- and event-level relations
    video_level_relations, _ = relations_lib.extract_relations(doc, timestamps, wn_dictionary, wn_lemmatizer)

    # print results
    print(f"--------------------------------------------------------------\n"
          f"Input: {text}\n")

    print("Detected Sentences:")
    for timestamp, sentence in zip(timestamps, sentences):
        print(f"{sentence}.")
    
    print("\nVideo-level Entities:")
    for e in video_level_entities:
        print(e.to_string())

    print("\nEntity-Property Pairs:")
    for ep in entity_property_pairs:
        print(ep.to_string())

    print("\nVideo-level Relations:")
    for r in video_level_relations:
        print(r.to_string())

    print()
