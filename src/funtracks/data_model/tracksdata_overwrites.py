from typing import Any

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph

def overwrite_graphview_add_node(
    self,
    attrs: dict[str, Any],
    validate_keys: bool = True,
    index: int | None = None,
) -> int | None:
    if index in self._root.node_ids():
        self._root.update_node_attrs(
            node_ids=[index],
            attrs={DEFAULT_ATTR_KEYS.SOLUTION: True},
        )
        parent_node_id = index
    else:
        with self._root.node_added.blocked():
            parent_node_id = self._root.add_node(
                attrs=attrs,
                validate_keys=validate_keys,
                index=index,
            )

    if self.sync:
        with self.node_added.blocked():
            node_id = RustWorkXGraph.add_node(
                self,
                attrs=attrs,
                validate_keys=validate_keys,
            )
        self._add_id_mapping(node_id, parent_node_id)
    else:
        self._out_of_sync = True

    self._root.node_added.emit_fast(parent_node_id)
    self.node_added.emit_fast(parent_node_id)

    return parent_node_id

def overwrite_graphview_add_edge(
    self,
    source_id: int,
    target_id: int,
    attrs: dict[str, Any],
    validate_keys: bool = True,
) -> int:
    
    if self._root.has_edge(source_id, target_id):
        self._root.update_edge_attrs(
            edge_ids=[self._root.edge_id(source_id, target_id)],
            attrs={DEFAULT_ATTR_KEYS.SOLUTION: True},
        )
        parent_edge_id = self._root.edge_id(source_id, target_id)
    else:
        parent_edge_id = self._root.add_edge(
            source_id=source_id,
            target_id=target_id,
            attrs=attrs,
            validate_keys=validate_keys,
        )
        attrs[DEFAULT_ATTR_KEYS.EDGE_ID] = parent_edge_id

    if self.sync:
        # it does not set the EDGE_ID as attribute as the super().add_edge
        edge_id = self.rx_graph.add_edge(
            self._map_to_local(source_id),
            self._map_to_local(target_id),
            attrs,
        )
        self._edge_map_to_root.put(edge_id, parent_edge_id)
    else:
        self._out_of_sync = True

    return parent_edge_id