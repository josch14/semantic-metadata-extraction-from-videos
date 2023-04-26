from abc import ABC, abstractmethod


class Entity(ABC):

    def __init__(self, name: str):
        assert isinstance(name, str), "entity name should be a string"

        self.name = name.lower()


    @abstractmethod
    def to_string(self) -> str:
        pass


    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
