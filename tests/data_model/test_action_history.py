from funtracks.data_model.action_history import ActionHistory
from funtracks.data_model.actions import AddNodes
from funtracks.data_model.tracks import Tracks
from funtracks.data_model.tracksdata_utils import create_empty_graphview_graph

# https://github.com/zaboople/klonk/blob/master/TheGURQ.md


def test_action_history():
    history = ActionHistory()

    # make an empty tracksdata graph with the default attributes
    graph_td = create_empty_graphview_graph(database=":memory:", position_attrs=["pos"])

    tracks = Tracks(graph_td, ndim=3)
    action1 = AddNodes(
        tracks, nodes=[0, 1], attributes={"t": [0, 1], "pos": [[0, 1], [1, 2]]}
    )

    # empty history has no undo or redo
    assert not history.undo()
    assert not history.redo()

    # add an action to the history
    history.add_new_action(action1)
    # undo the action
    assert history.undo()
    assert tracks.graph.num_nodes == 0
    assert len(history.undo_stack) == 1
    assert len(history.redo_stack) == 1
    assert history._undo_pointer == -1

    # no more actions to undo
    assert not history.undo()

    # redo the action
    assert history.redo()
    assert tracks.graph.num_nodes == 2
    assert len(history.undo_stack) == 1
    assert len(history.redo_stack) == 0
    assert history._undo_pointer == 0

    # no more actions to redo
    assert not history.redo()

    # undo and then add new action
    assert history.undo()
    action2 = AddNodes(tracks, nodes=[10], attributes={"t": [10], "pos": [[0, 1]]})
    history.add_new_action(action2)
    assert tracks.graph.num_nodes == 1
    # there are 3 things on the stack: action1, action1's inverse, and action 2
    assert len(history.undo_stack) == 3
    assert len(history.redo_stack) == 0
    assert history._undo_pointer == 2

    # undo back to after action 1
    assert history.undo()
    assert history.undo()
    assert tracks.graph.num_nodes == 2

    assert len(history.undo_stack) == 3
    assert len(history.redo_stack) == 2
    assert history._undo_pointer == 0
