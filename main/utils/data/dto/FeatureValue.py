from typing import List

from pandas import DataFrame

from main.utils.data.dto.FeatureAnalysis import FeatureAnalysis



class FeatureValue:
    def __init__(self, name: str, type: str, subs: List['FeatureValue'] = None):
        self.name = name
        self.type = type
        self.subs = subs if subs is not None else []


    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureValue':
        """Create a Group object from a dictionary."""
        subs = [cls.from_dict(subs) for subs in
                data.get("subs", [])]  # Recursively convert subs
        return cls(name=data["title"], type=data["type"], subs=subs)

    @classmethod
    def from_json(cls, data) -> List['FeatureValue']:
        """Create a list of Group objects from a JSON string."""
        return [cls.from_dict(item) for item in data]

    def __repr__(self):
        return f"RawRowValue(name={self.name}, type={self.type}, subs={self.subs})"

    def to_dict(self):
        return {
            
            "name": self.name,
            "type": self.type,
            "subs":  [x.to_dict() for x in self.subs],
        }

