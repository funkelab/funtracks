# GEFF Import Process Flow

This diagram shows the steps involved in importing tracking data from GEFF format and the dependencies between them.

```mermaid
graph TD
    Start([Start Import]) --> ImportGraph[import_graph_from_geff<br/>→ InMemoryGeff with standard keys]
    Start --> LoadSeg[Lazily load segmentation<br/>→ dask seg array]

    ImportGraph --> ValidateIDs[Validate track_ids & lineage_ids<br/>→ validation results]
    ImportGraph --> CheckRelabel{validate_graph_seg_match<br/>Check if segmentation<br/>needs relabeling?}
    LoadSeg --> CheckRelabel

    CheckRelabel -->|needs relabeling| RelabelSeg[Relabel segmentation<br/>seg_id → node_id]
    CheckRelabel -->|already labeled| SkipRelabel[Use segmentation as-is]

    RelabelSeg --> SegReady[Segmentation ready]
    SkipRelabel --> SegReady

    ValidateIDs --> ConstructGraph[Construct NetworkX graph<br/>from InMemoryGeff]
    ImportGraph --> ConstructGraph

    ConstructGraph --> ComputeFeatures[Compute features<br/>using annotators]
    SegReady -.->|if needed| ComputeFeatures

    ComputeFeatures --> RegisterFeatures[Register features<br/>in FeatureDict]

    RegisterFeatures --> CreateTracks[Create SolutionTracks]
    ConstructGraph --> CreateTracks
    SegReady -.->|optional| CreateTracks

    CreateTracks --> End([Return SolutionTracks])

    style ImportGraph fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style LoadSeg fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style CheckRelabel fill:#E94B8B,stroke:#B83A6D,stroke-width:3px,color:#fff
    style RelabelSeg fill:#E94B8B,stroke:#B83A6D,stroke-width:3px,color:#fff
    style ValidateIDs fill:#F5A623,stroke:#C77D1A,stroke-width:3px,color:#fff
    style ConstructGraph fill:#50C878,stroke:#3A9B5C,stroke-width:3px,color:#fff
    style ComputeFeatures fill:#50C878,stroke:#3A9B5C,stroke-width:3px,color:#fff
    style RegisterFeatures fill:#50C878,stroke:#3A9B5C,stroke-width:3px,color:#fff
    style CreateTracks fill:#E85D75,stroke:#B84A5F,stroke-width:3px,color:#fff
```

## Process Steps

### 1. Loading Phase (Blue)
**Parallel operations** - can happen independently:
- **`import_graph_from_geff()`**: Loads GEFF data and renames all property keys from custom to standard
  - Reads GEFF data into `InMemoryGeff` format
  - Transforms custom property names to standard keys (e.g., `"t"` → `"time"`, `"circ"` → `"circularity"`)
  - Returns InMemoryGeff with standard keys
- **Load segmentation from disk**: Reads segmentation array from file (tif, zarr, etc.)

### 2. Validation Phase (Yellow)
**Sequential operations** on the loaded data:
- **Validate track_ids & lineage_ids**: Check if provided IDs are valid according to GEFF spec

### 3. Segmentation Validation Phase (Pink)
**Conditional operations** on segmentation:
- **Check if relabeling needed**: Compares segmentation IDs with node IDs
- **Relabel segmentation** (if needed): Maps seg_id values to node_id values
- **Use as-is** (if not needed): Segmentation already uses node IDs

### 4. Construction Phase (Green)
**Build the final data structures**:
- **Construct NetworkX graph**: Create graph from InMemoryGeff (now with standard keys)
- **Compute features**: Run annotators to compute features (may use segmentation)
- **Register features**: Add computed features to FeatureDict

### 5. Final Assembly (Red)
- **Create SolutionTracks**: Assemble all components into final Tracks object

## Key Dependencies

### Critical Path
The longest dependency chain:
```
import_graph_from_geff → Validate IDs → Construct graph →
Compute features → Register features → Create SolutionTracks
```

### Parallel Opportunities
- `import_graph_from_geff()` and segmentation loading are independent
- Segmentation validation can happen in parallel with track_id validation

### Design Insight
**Why `import_graph_from_geff()` returns standard keys:**

By renaming property keys in the InMemoryGeff *before* constructing the NetworkX graph:
1. Graph is created with standard keys from the start
2. No need for post-construction key renaming
3. All downstream code (validation, feature computation) works with standard keys
4. Single transformation point - simpler and more maintainable
