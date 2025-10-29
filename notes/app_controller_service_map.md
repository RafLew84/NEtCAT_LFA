# AppController Service Extraction Map

## 1. History Orchestration & Session Lifecycle
- `save_analysis_session`, `load_analysis_session`
- `delete_history_step`, `delete_original_image`, `_handle_history_deletion_result`
- `reset_session`, `load_file`, `load_metadata_into_session`
- `add_operation_to_history`, `add_new_node_to_history`
- `get_active_history_node`, `get_current_image_data_for_processing`, `get_current_node_info_for_dialogs`
- `_on_history_active_node_changed` (event bridge)
- `export_session_state`, `load_session_state`

### Candidate Service: `HistoryOrchestrator`
Responsibilities:
- Keep HistoryManager in sync with controller actions.
- Provide helpers for node lookup and deletion flows.
- Mediate session (re)loading and ensure selection state is consistent.

## 2. Spot & Lattice Management
- Selection / mode: `set_spot_selection_mode`, `set_spot_refinement_method`, `set_refinement_roi_size`
- Visibility toggles: `set_substrate_raw_visibility`, `set_substrate_transformed_visibility`, `set_adsorbate_raw_visibility`, `set_adsorbate_transformed_visibility`
- Spot set operations: `clear_last_adsorbate_spot`, `reselect_current_adsorbate_set`, `clear_all_adsorbate_sets`, `add_new_adsorbate_set`, `set_current_adsorbate_set_by_index`, `clear_all_spot_data`
- Result updates: `update_substrate_analysis_results`, `update_adsorbate_set_results`, `update_superstructure_periodicity_results`
- Lattice types/offsets: `set_expected_adsorbate_lattice_type`, `reference_ideal_substrate_spots_px` management
- Real-space calculations: `calculate_and_store_substrate_real_params`, `calculate_and_store_adsorbate_real_params`
- FFT panel evaluation: `evaluate_fft_panel_state`, `_has_valid_substrate_definition`, `_get_fft_data_shape`, `_can_calculate_substrate_real_space`, `_can_calculate_adsorbate_real_space`

### Candidate Service: `SpotSetService` / `LatticeAnalysisService`
Responsibilities:
- Own raw/corrected spot lists, expected lattice types, visibility flags.
- Offer DTOs for UI (panel state, overlays) separate from controller.
- Trigger recalculations and emit appropriate signals.

## 3. Preprocessing & Analysis Execution
- Preprocessing operations: `apply_gaussian_blur`, `apply_gaussian_sharpening`, `apply_plane_leveling`, `apply_median_filter`, `apply_nlmeans_denoising`, `apply_bm3d_denoising`
- FFT/STMs: `calculate_fft_operation`, `apply_stm_transform`
- File analysis helpers: `calculate_and_store_*` (listed above)
- Convenience `can_*` guards: `can_calculate_fft`, `can_select_spots`, `can_analyze_superstructure`, `can_visualize_real_space`, `can_open_real_space_reconstruction`, `can_open_stm_transform`, `can_load_metadata`

### Candidate Service: `AnalysisExecutor`
Responsibilities:
- Wrap preprocessing pipelines and FFT calculations.
- Guard availability checks and provide status objects for UI gating.
- Handle failure reporting / signal emission on success.

## 4. Session I/O & Persistence
- `save_analysis_session`, `load_analysis_session` (listed above)
- `export_session_state`, `load_session_state`
- Interaction with `SessionSerializer`, disk operations in `load_file`

### Candidate Service: `SessionService`
Responsibilities:
- Marshal controller state to/from persistent formats.
- Coordinate file picker interactions (optionally via facade to MainWindow).
- House version migration awareness distinct from application logic.

## Cross-cutting Considerations
- Signal emissions (e.g., `spot_lists_updated`, `substrate_transform_results_updated`) should move into respective services, potentially via event buses.
- Shared data classes (e.g., FFT parameters, real-space DTOs) should live in `lfa/core` to decouple services from Qt.
- Services need access to HistoryManager and VisualizationManager; prefer dependency injection via constructor.

## Proposed Extraction Order
1. Session-oriented logic (History + Session) — minimal UI coupling, high cohesion.
2. Spot/Lattice management — already partly delegated to `SpotManager`; extend or replace with richer service.
3. Analysis execution — separate heavy processing functions from signal bookkeeping.
