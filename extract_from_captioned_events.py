from src import nlp_lib
from src.entities_lib import EntitiesLib
from src.relations_lib import RelationsLib
from src.wordnet_lib import WordNetDictionary, WordNetLemmatizerWrapped

EXAMPLES = []

# example 1
sentences = [
    "A man and a dog walk onto a wide field.",
    "The man throws a red frisbee and the dog chases after it.",
    "The dog brings the frisbee back to the man.",
    "The whole time there are people on the sidelines watching them and taking pictures."]
timestamps = [[0.26, 12.22], [12.76, 36.27], [30.22, 46.00], [0.45, 46.23]]
EXAMPLES.append((sentences, timestamps))

# example 2
sentences = ["A girl is seen dribbling with a football.", "She then kicks it at a goal."]
timestamps = [[3.20, 10.11], [12.05, 16.40]]
EXAMPLES.append((sentences, timestamps))



if __name__ == "__main__":

    # load WordNet
    wn_dictionary = WordNetDictionary()
    wn_lemmatizer = WordNetLemmatizerWrapped()

    for sentences, timestamps in EXAMPLES:
        """
        Event Processing.
        """
        # sort sentences according to their starting times
        starting_times = [t[0] for t in timestamps]
        starting_times, timestamps, sentences = (list(t) for t in zip(*sorted(zip(starting_times, timestamps, sentences))))

        # create linguisic annoations with the language parser
        doc = nlp_lib.parse(sentences)


        # extract Semantic Metadata
        semantic_metadata = {}
        """
        1) Extract video- and event-level entities and entity-property pairs.
        """
        video_level_entities, event_level_entities, entity_property_pairs = \
            EntitiesLib.extract_entities_and_properties(doc, timestamps, wn_dictionary, wn_lemmatizer)


        """
        2) Extract video- and event-level relations.
        """
        video_level_relations, event_level_relations = \
            RelationsLib.extract_relations(doc, timestamps, wn_dictionary, wn_lemmatizer)
    
        print("--------------------------------------------------------------")
        print("Timestamps \& Sentences:")
        for timestamp, sentence in zip(timestamps, sentences):
            print(f"{timestamp}: {sentence}")
        
        print("\nVideo-level Entities:")
        for e in video_level_entities:
            print(e.to_string())

        print("\nEntity-Property Pairs:")
        for ep in entity_property_pairs:
            print(ep.to_string())

        print("\nVideo-level Relations:")
        for r in video_level_relations:
            print(r.to_string())

        print("\nEvent-level Entities:")
        for e in event_level_entities:
            print(e.to_string())

        print("\nEvent-level Relations:")
        for r in event_level_relations:
            print(r.to_string())

        print()
