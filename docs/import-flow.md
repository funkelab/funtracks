# GEFF Import Process Flow

This diagram shows the steps involved in importing tracking data from GEFF format and the dependencies between them.

```mermaid
graph TD
    Start([Start Import]) --> ImportGraph[import_graph_from_geff<br/>→ InMemoryGeff with standard keys]
    Start --> LoadSeg[Lazily load segmentation<br/>→ dask seg array]

    ImportGraph --> ValidateIDs[Validate track_ids & lineage_ids<br/>→ remove if invalid]

    ValidateIDs --> ConstructGraph[Construct NetworkX graph<br/>from InMemoryGeff]
    ImportGraph --> ConstructGraph

    ConstructGraph --> ValidateGEFF{Segmentation<br/>provided?}
    LoadSeg --> ValidateGEFF

    ValidateGEFF -->|Yes| GEFFChecks[GEFF validation:<br/>axes_match_seg_dims<br/>has_valid_seg_id]
    ValidateGEFF -->|No| CreateTracks

    GEFFChecks --> CheckRelabel{validate_graph_seg_match<br/>Check if segmentation<br/>needs relabeling?}

    CheckRelabel -->|needs relabeling| RelabelSeg[Relabel segmentation<br/>seg_id → node_id]
    CheckRelabel -->|already labeled| SkipRelabel[Use segmentation as-is]

    RelabelSeg --> SegReady[Segmentation ready]
    SkipRelabel --> SegReady

    SegReady --> CreateTracks[Create SolutionTracks<br/>with graph & segmentation]

    CreateTracks --> ValidateFeatures{node_features<br/>requested?}

    ValidateFeatures -->|Yes| CheckFeatures[Validate features exist<br/>in annotators or GEFF]
    ValidateFeatures -->|No| End

    CheckFeatures --> EnableFeatures[enable_features with<br/>recompute flag for each]

    EnableFeatures --> RegisterStatic[Register static features<br/>not in annotators]

    RegisterStatic --> End([Return SolutionTracks])

    style ImportGraph fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style LoadSeg fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style ValidateGEFF fill:#F5A623,stroke:#C77D1A,stroke-width:3px,color:#fff
    style GEFFChecks fill:#F5A623,stroke:#C77D1A,stroke-width:3px,color:#fff
    style CheckRelabel fill:#E94B8B,stroke:#B83A6D,stroke-width:3px,color:#fff
    style RelabelSeg fill:#E94B8B,stroke:#B83A6D,stroke-width:3px,color:#fff
    style ValidateIDs fill:#F5A623,stroke:#C77D1A,stroke-width:3px,color:#fff
    style ConstructGraph fill:#50C878,stroke:#3A9B5C,stroke-width:3px,color:#fff
    style CreateTracks fill:#E85D75,stroke:#B84A5F,stroke-width:3px,color:#fff
    style ValidateFeatures fill:#F5A623,stroke:#C77D1A,stroke-width:3px,color:#fff
    style CheckFeatures fill:#F5A623,stroke:#C77D1A,stroke-width:3px,color:#fff
    style EnableFeatures fill:#50C878,stroke:#3A9B5C,stroke-width:3px,color:#fff
    style RegisterStatic fill:#50C878,stroke:#3A9B5C,stroke-width:3px,color:#fff
```

## Process Steps

### 1. Loading Phase (Blue)
**Parallel operations** - can happen independently:
- **`import_graph_from_geff()`**: Loads GEFF data and renames all property keys from custom to standard
  - Reads GEFF data into `InMemoryGeff` format
  - Transforms custom property names to standard keys (e.g., `"t"` → `"time"`, `"circ"` → `"circularity"`)
  - Returns InMemoryGeff with standard keys
- **Load segmentation from disk**: Lazily reads segmentation array from file (tif, zarr, etc.) into dask array

### 2. Validation Phase (Yellow)
**Sequential operations** on the loaded data:
- **Validate track_ids & lineage_ids**: Check if provided IDs are valid according to GEFF spec, remove if invalid
- **GEFF-specific validation** (if segmentation provided):
  - **`axes_match_seg_dims`**: Check axes metadata matches segmentation dimensions
  - **`has_valid_seg_id`**: Validate seg_ids are integers
- **Feature validation** (if features requested):
  - Features with `recompute=True` must exist in annotators
  - Features with `recompute=False` must exist in GEFF node_props

### 3. Construction Phase (Green)
**Build the graph and tracks**:
- **Construct NetworkX graph**: Create graph from InMemoryGeff (now with standard keys)
- **Validate & relabel segmentation** (if provided):
  - Check if relabeling needed using generic `validate_graph_seg_match`
  - Relabel if seg_id ≠ node_id
- **Create SolutionTracks**: Assemble graph and segmentation into Tracks object

### 4. Segmentation Validation Phase (Pink)
**Conditional operations** on segmentation (happens after graph construction):
- **Check if relabeling needed**: Uses NetworkX graph to compare segmentation IDs with node IDs
- **Relabel segmentation** (if needed): Maps seg_id values to node_id values
- **Use as-is** (if not needed): Segmentation already uses node IDs

### 5. Feature Registration Phase (Green)
**Happens AFTER SolutionTracks creation** - if `node_features` provided:
- **Enable features**: Call `tracks.enable_features(key, recompute=flag)` for each feature
  - If `recompute=True`: Activates annotator and computes values
  - If `recompute=False`: Activates annotator but uses existing values from graph
- **Register static features**: Add non-annotator features to FeatureDict

### 6. Final Return (Red)
- **Return SolutionTracks**: Return fully configured Tracks object with features enabled

## Key Dependencies

### Critical Path
The longest dependency chain:
```
import_graph_from_geff → Validate IDs → Construct graph →
[if segmentation] GEFF validation → Validate match → Relabel (if needed) →
Create SolutionTracks → [if features] Validate & enable features
```

### Parallel Opportunities
- `import_graph_from_geff()` and segmentation loading are independent and can happen in parallel

### Design Insights

**Why `import_graph_from_geff()` returns standard keys:**
By renaming property keys in the InMemoryGeff *before* constructing the NetworkX graph:
1. Graph is created with standard keys from the start
2. No need for post-construction key renaming
3. All downstream code (validation, feature computation) works with standard keys
4. Single transformation point - simpler and more maintainable

**Why features are enabled AFTER SolutionTracks creation:**
1. Tracks object must exist before features can be computed or registered
2. Feature computation may require the tracks object (e.g., for segmentation access)
3. Separation of concerns: graph/segmentation construction vs. feature management
4. Allows validation of feature requests against available annotators
