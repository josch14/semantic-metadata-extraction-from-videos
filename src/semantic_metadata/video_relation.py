from src.semantic_metadata.relation import Relation
from src.wordnet_lib import WordNetLemmatizerWrapped, get_words_from_synsets


class VideoRelation(Relation):


    def __init__(self, subjects: list, verb: str, modifiers: list, objects: list):
        super().__init__(subjects, verb, modifiers, objects)


    @staticmethod
    def from_dict(d: dict):
        video_relation = VideoRelation(
            subjects=d["s"],
            verb=d["v"],
            modifiers=d["m"],
            objects=d["o"],
        )

        return video_relation


    def to_string(self) -> str:
        subjects_str = self.subjects if len(self.subjects) > 1 else self.subjects[0]
        objects_str = self.objects if len(self.objects) > 1 else self.objects[0]

        return f"({subjects_str}, {self.verb}, {self.modifiers}, {objects_str})"


    def to_dict(self) -> dict:
        d = {
            's': self.subjects,
            'v': self.verb,
            'm': self.modifiers,
            'o': self.objects
        }

        return d


    def __eq__(self, other) -> bool:
        if not isinstance(other, VideoRelation):
            return False
        if not self.subjects == other.subjects:  # subjects are assumed to be sorted
            return False
        if not self.verb == other.verb:
            return False
        if not self.modifiers == other.modifiers:
            return False
        if not self.objects == other.objects:
            return False

        return True


    def predicts(self, gt, wn_lemmatizer: WordNetLemmatizerWrapped, with_synonyms: bool = True) -> bool:
        assert isinstance(gt, VideoRelation), "given ground truth is not an video-level relation"

        # Check subjects predictions
        if not predicts_subject_or_object(self.subjects, gt.subjects, wn_lemmatizer, with_synonyms):
            return False
        if not predicts_verb(self.verb, self.modifiers, gt.verb, gt.modifiers, wn_lemmatizer, with_synonyms):
            return False
        if not predicts_subject_or_object(self.objects, gt.objects, wn_lemmatizer, with_synonyms):
            return False

        return True


def predicts_subject_or_object(predictions: list,
                               gt: list,
                               wn_lemmatizer: WordNetLemmatizerWrapped,
                               with_synonyms: bool = True
                               ) -> bool:
    """
    subject: e.g. self=["man"] predicts gt_relation=["men", "woman"]
    Prediction is correct if the prediction equals a ground truth entity or any synonym of such
    """

    # lemmatize prediction and gt entities
    predictions = [wn_lemmatizer.lemmatize_noun(e) for e in predictions]
    gt = [wn_lemmatizer.lemmatize_noun(e) for e in gt]

    gt_list = []
    for entity in gt:
        gt_list += [entity]
        # if desired, add all synonyms of a gt entity
        if with_synonyms:
            gt_list += get_words_from_synsets(entity, "noun")

    for p in predictions:
        if p in gt_list:
            return True

    return False


def predicts_verb(prediction_verb: str,
                  prediction_modifiers: list,
                  gt_verb: str,
                  gt_modifiers: list,
                  wn_lemmatizer: WordNetLemmatizerWrapped,
                  with_synonyms: bool = True
                  ) -> bool:

    # use verb and modifiers in all combinations
    prediction_verb = wn_lemmatizer.lemmatize_verb(prediction_verb)
    gt_verb = wn_lemmatizer.lemmatize_verb(gt_verb)
    prediction = [prediction_verb] + [(prediction_verb + "_" + m) for m in prediction_modifiers]
    gt = [gt_verb] + [(gt_verb + "_" + m) for m in gt_modifiers]


    # now for gt verbs: add all synsets
    gt_list = []
    for verb in gt:
        gt_list += [verb]
        # if desired, add all synonyms of a gt verb
        if with_synonyms:
            gt_list += get_words_from_synsets(verb, "verb")

    for p in prediction:
        if p in gt_list:
            return True

    return False
