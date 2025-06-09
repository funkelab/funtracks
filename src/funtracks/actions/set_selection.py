from ..project import Project
from .set_feature_values import SetFeatureValues


class SetSelection(SetFeatureValues):
    def __init__(
        self,
        project: Project,
        element: int | tuple[int, int],
        value: bool = True,
    ):
        graph_features = project.cand_graph.features
        if isinstance(element, int):
            features = {graph_features.node_selected: value}
        else:
            features = {graph_features.edge_selected: value}
        self.value = value
        super().__init__(project, element, features)
        # TODO: Make a feature type that is both node and edge?

    def inverse(self):
        """Delete edges"""
        return SetSelection(self.project, self.element, value=not self.value)

    def _apply(self):
        super()._apply()
        if self.value is True:
            if isinstance(self.element, int):
                self.project.solution.add_node(self.element)
            else:
                self.project.solution.add_edge(self.element)
        elif self.value is False:
            if isinstance(self.element, int):
                self.project.solution.remove_node(self.element)
            else:
                self.project.solution.remove_edge(self.element)
