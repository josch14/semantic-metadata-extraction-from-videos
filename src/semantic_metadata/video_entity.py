from src.semantic_metadata.entity import Entity
from src.wordnet_lib import WordNetLemmatizerWrapped, get_words_from_synsets


class VideoEntity(Entity):

    def __init__(self, name):
        super().__init__(name)


    @staticmethod
    def from_dict(d: dict):
        video_entity = VideoEntity(
            name=d["n"],
        )

        return video_entity


    def to_string(self) -> str:
        return f"{self.name}"


    def to_dict(self) -> dict:
        d = {
            'n': self.name
        }

        return d


    def __eq__(self, other) -> bool:
        if not isinstance(other, VideoEntity):
            return False
        if not self.name == other.name:
            return False

        return True


    def predicts(self, gt, wn_lemmatizer: WordNetLemmatizerWrapped, with_synonyms: bool = True) -> bool:
        assert isinstance(gt, VideoEntity), "given ground truth is not an video-level entity"

        prediction_name = wn_lemmatizer.lemmatize_noun(self.name)
        gt_name = wn_lemmatizer.lemmatize_noun(gt.name)

        # if desired, add all synonyms of gt entities
        gt_list = [gt_name] if not with_synonyms else \
            [gt_name] + get_words_from_synsets(gt_name, "noun")

        if prediction_name not in gt_list:
            return False

        return True
