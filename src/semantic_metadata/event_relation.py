from src.semantic_metadata.relation import Relation
from src.semantic_metadata.video_relation import VideoRelation
from src.utils import calculate_iou
from src.wordnet_lib import WordNetLemmatizerWrapped


class EventRelation(Relation):

    def __init__(self, subjects: list, verb: str, modifiers: list, objects: list, timestamp: list):
        super().__init__(subjects, verb, modifiers, objects)

        self.__check_timestamp(timestamp)
        self.timestamp = timestamp


    @staticmethod
    def from_dict(d: dict):
        event_relation = EventRelation(
            subjects=d["s"],
            verb=d["v"],
            modifiers=d["m"],
            objects=d["o"],
            timestamp=d["t"]
        )

        return event_relation


    @staticmethod
    def __check_timestamp(timestamp: list):
        assert isinstance(timestamp, list) and len(timestamp) == 2, \
            f"event-level relation could not be initialized: timestamp {timestamp} in unexpected format"


    def to_string(self) -> str:
        subjects_str = self.subjects if len(self.subjects) > 1 else self.subjects[0]
        objects_str = self.objects if len(self.objects) > 1 else self.objects[0]

        return f"{self.timestamp}: ({subjects_str}, {self.verb}, {self.modifiers}, {objects_str})"


    def to_dict(self) -> dict:
        d = {
            't': self.timestamp,
            's': self.subjects,
            'v': self.verb,
            'm': self.modifiers,
            'o': self.objects
        }

        return d


    def __eq__(self, other) -> bool:
        if not isinstance(other, EventRelation):
            return False
        if not self.subjects == other.subjects:  # subjects are assumed to be sorted
            return False
        if not self.verb == other.verb:
            return False
        if not self.modifiers == other.modifiers:
            return False
        if not self.objects == other.objects:
            return False
        if not self.timestamp == other.timestamp:
            return False

        return True


    def predicts(self, gt, target_iou: float, wn_lemmatizer: WordNetLemmatizerWrapped, with_synonyms=True) -> bool:
        assert isinstance(gt, EventRelation), "given ground truth is not an event-level relation"

        if calculate_iou(self.timestamp, gt.timestamp) < target_iou:
            return False

        # IoU is fine, therefore check remaining elements using the corresponding video-level method
        is_prediction = VideoRelation(self.subjects, self.verb, self.modifiers, self.objects).predicts(
            VideoRelation(gt.subjects, gt.verb, gt.modifiers, gt.objects), wn_lemmatizer, with_synonyms)

        return is_prediction
