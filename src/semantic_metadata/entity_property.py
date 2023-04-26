from src.semantic_metadata.video_entity import VideoEntity
from src.wordnet_lib import get_words_from_synsets


class EntityPropertyPair:

    def __init__(self, entity: str, property: str):
        assert isinstance(entity, str), "entity name should be a string"
        assert isinstance(property, str), "property name should be a string"

        self.entity = entity.lower()
        self.property = property.lower()  # properties are lemmatized already


    @staticmethod
    def from_dict(d: dict):
        event_relation = EntityPropertyPair(
            entity=d["e"],
            property=d["p"]
        )

        return event_relation


    def to_string(self) -> str:
        return f"{self.entity} [{self.property}]"


    def to_dict(self) -> dict:
        d = {
            'e': self.entity,
            'p': self.property
        }

        return d


    def __eq__(self, other) -> bool:
        if not isinstance(other, EntityPropertyPair):
            return False
        if not self.entity == other.entity:
            return False
        if not self.property == other.property:
            return False

        return True


    def predicts(self, gt, with_synonyms=True) -> bool:
        assert isinstance(gt, EntityPropertyPair), "given ground truth is not an entity-property pair"

        # compare entity
        s_entity = VideoEntity(self.entity)
        gt_entity = VideoEntity(gt.entity)
        if not s_entity.predicts(gt_entity):
            return False

        # compare property (properties are defined lemmatized -> no lemmatization necessary)
        gt_list = [gt.property] if not with_synonyms else \
            [gt.property] + get_words_from_synsets(gt.property, "noun")
        if self.property not in gt_list:
            return False

        return True
