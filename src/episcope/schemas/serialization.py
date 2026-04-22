import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class Serializable(ABC):
    """Base class for serializable data structures."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create object from dictionary representation."""
        pass

    def to_json(self, filepath: str) -> None:
        """Save object to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str) -> 'Serializable':
        """Load object from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class SerializableList(Serializable):
    """Base class for lists of serializable objects."""

    def __init__(self, items=None):
        self.items = items or []

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SerializableList':
        # This will be overridden in subclasses with specific item types
        items = []
        for item_data in data.get("items", []):
            items.append(cls._item_class().from_dict(item_data))
        return cls(items)

    @classmethod
    def _item_class(cls):
        """Override in subclasses to specify the item type."""
        raise NotImplementedError("Subclasses must implement _item_class")

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]
