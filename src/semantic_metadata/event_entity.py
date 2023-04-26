from src.semantic_metadata.entity import Entity
from src.semantic_metadata.video_entity import VideoEntity
from src.utils import calculate_iou
from src.wordnet_lib import WordNetLemmatizerWrapped


class EventEntity(Entity):

    def __init__(self, name, timestamp):
        super().__init__(name)

        self.__check_timestamp(timestamp)
        self.timestamp = timestamp


    @staticmethod
    def from_dict(d: dict):
        event_entity = EventEntity(
            name=d["n"],
            timestamp=d["t"]
        )

        return event_entity


    @staticmethod
    def __check_timestamp(timestamp: list):
        assert isinstance(timestamp, list) and len(timestamp) == 2, \
            f"event-level relation could not be initialized: timestamp {timestamp} in unexpected format"


    def to_string(self) -> str:
        return f"{self.timestamp}: {self.name}"


    def to_dict(self) -> dict:
        d = {
            't': self.timestamp,
            'n': self.name
        }

        return d


    def __eq__(self, other) -> bool:
        if not isinstance(other, EventEntity):
            return False
        if not self.name == other.name:
            # entity name is compared non-lemmatized. If the name should be compared lemmatized, this has to be
            # specified before the method call (e.g., during evaluation)
            return False
        if not self.timestamp == other.timestamp:
            return False

        return True


    def predicts(self, gt, target_iou: float, wn_lemmatizer: WordNetLemmatizerWrapped, with_synonyms: bool = True) -> bool:
        assert isinstance(gt, EventEntity), "given ground truth is not an event-level entity"

        if calculate_iou(self.timestamp, gt.timestamp) < target_iou:
            return False

        # IoU is fine, therefore check remaining elements using the corresponding video-level method
        is_prediction = VideoEntity(self.name).predicts(
            VideoEntity(gt.name), wn_lemmatizer, with_synonyms
        )

        return is_prediction
