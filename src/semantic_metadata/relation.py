from abc import ABC, abstractmethod


class Relation(ABC):

    def __init__(self, subjects: list, verb: str, modifiers: list, objects: list):

        # check appropriate definition of relation elements
        self.__check_string_list(subjects, check_len_greater_zero=True)
        assert isinstance(verb, str), "verb should be a string"
        self.__check_string_list(modifiers, check_len_greater_zero=False)
        self.__check_string_list(objects, check_len_greater_zero=True)

        # set attributes
        self.subjects = [s.lower() for s in subjects]
        self.verb = verb.lower()
        self.modifiers = [w.lower() for w in modifiers]
        self.objects = [o.lower() for o in objects]

        # sort subjects and objects (required for equals-methods of EventRelation and VideoRelation)
        self.subjects.sort()
        self.objects.sort()


    @staticmethod
    def __check_string_list(input_list, check_len_greater_zero: bool):
        assert isinstance(input_list, list), "relation could not be initialized: no string list"

        if check_len_greater_zero:
            assert len(input_list) > 0, "relation could not be initialized: string list should not be empty"

        for e in input_list:
            assert isinstance(e, str), "relation could not be initialized: no string list"


    @abstractmethod
    def to_string(self) -> str:
        pass


    @abstractmethod
    def to_dict(self) -> dict:
        pass


    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
