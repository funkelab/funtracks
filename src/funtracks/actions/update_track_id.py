from ..project import Project
from ._base import TracksAction


class UpdateTrackID(TracksAction):
    def __init__(self, project: Project, start_node: int, track_id: int):
        """
        Args:
            project (Project):
            start_node (Node): The node ID of the first node in the track. All successors
                with the same track id as this node will be updated.
            track_id (int): The new track id to assign.
        """
        super().__init__(project)
        self.start_node = start_node
        self.old_track_id = self.project.cand_graph.get_track_id(self.start_node)
        self.new_track_id = track_id
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore the previous track_id"""
        return UpdateTrackID(self.project, self.start_node, self.old_track_id)

    def _apply(self):
        """Assign a new track id to the track starting with start_node."""
        curr_node = self.start_node
        while self.project.cand_graph.get_track_id(curr_node) == self.old_track_id:
            # update the track id
            self.project.cand_graph.set_track_id(curr_node, self.new_track_id)
            # getting the next node (picks one if there are two)
            successors = list(self.project.solution.successors(curr_node))
            if len(successors) == 0:
                break
            curr_node = successors[0]
