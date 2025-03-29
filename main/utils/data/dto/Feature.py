class IndependentFeature:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"IndependentFeature(name={self.name})"
    def to_dict(self):
        return {
            
            "name": self.name,
        }

class DependentFeature(IndependentFeature):
    def __init__(self, name: str, children: list['DependentFeature'] = None):
        super().__init__(name)
        self.name = name
        self.path = name.lower()
        self.children = children
        for x in self.children or []:
            x.path = f'{self.path}|{x.path}'
    def __repr__(self):
        return f"DependentFeature(name={self.name},child={self.children})"
    def to_dict(self):
        return {
            
            "name": self.name,
            "children": self.children,
        }

