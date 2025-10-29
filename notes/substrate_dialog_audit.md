# Substrate Spot Dialog – Responsibility Audit (Milestone 3.1)

## Current Responsibilities (single class)
- **Qt View Construction**: builds the entire widget tree (splitters, lists, spin boxes, buttons), populates combos and handles visibility toggles.
- **State Management**: stores selected spots, lattice selections, custom parameters, fitted spots, transformation matrices, last preview data, ROI size/method etc.
- **PyQtGraph Scene Control**: creates/updates `ImageItem`, ROI, scatter plot overlays, redraws markers after every action, handles mouse clicks.
- **History/Domain Access**: reaches into `HistoryManager` to fetch original nodes, calibration data, display names; mutates controller state via returned results.
- **Lattice/Transform Logic**: gathers lattice info, generates reciprocal points, calls `match_and_fit_transform`, applies affine transforms, builds analysis summaries.
- **Spot Refinement Logic**: performs ROI sampling, max-pixel lookup, Gaussian fitting, ROI preview caching.
- **Validation & Messaging**: enforces spot limits, displays warnings/errors, formats results via `format_float/format_pair`.
- **Result Packaging**: composes dictionary returned to controller (`spots`, `transform`, `custom_definition` etc.).

## Pain Points
- Mixing UI code with heavy numeric processing, making testing difficult.
- Direct dependency on `HistoryManager` and analysis modules from the dialog (tight coupling).
- All state transitions happen via mutable attributes; no separation between view state and domain state.
- Repeated redraw/update logic scattered across multiple methods.
- Hard to reuse transform/lattice calculation elsewhere (e.g. presenters or headless workflows).

## Proposed Split
1. **View (Qt dialog subclass)**  
   - Own widget creation, signal wiring, simple view updates (enabling buttons, setting labels).  
   - Delegate user actions (clicks, ROI changes, combo selection) to a presenter/viewmodel.  
   - Render overlays via an adapter but receive ready-to-draw data from the presenter.

2. **Presenter / ViewModel (`SubstrateSpotPresenter`)**  
   - Hold spot lists, lattice selection, transform results.  
   - Expose intent methods (`add_spot_from_roi`, `calculate_transform`, `change_lattice`, `clear_spots`, etc.).  
   - Communicate with services (`HistoryOrchestrator`, future `SpotSetService`, drift correction utilities).  
   - Produce DTOs for the view: overlay specs, status strings, button enablement flags, result payload.

3. **Scene/Overlay Adapter (`SubstrateSpotScene`)**  
   - Responsible solely for pyqtgraph items (ROI, scatter plots).  
   - Accept commands like `show_spots`, `show_fitted_spots`, `set_roi_state`.  
   - Presenter manipulates data; scene updates visuals.

4. **Utility Modules**  
   - `lfa/analysis/substrate_transform.py` (wrap `match_and_fit_transform`, `analyze_affine_transform`; manage ideal point generation).  
   - `lfa/analysis/spot_refinement.py` (ROI sampling, Gaussian preview).  
   - Shared dataclasses for dialog state/result.

## Immediate Extraction Targets
- `_on_calculate_transform_clicked` & `_build_lattice_info_dict` → transform utility/presenter.
- `_add_current_roi_spot`, ROI preview functions → spot refinement helper.
- History lookups (`get_root_node_for_node`, `get_next_original_display_name`) → service calls (already started via `HistoryOrchestrator`).  
- `selected_spots`, `current_lattice_type`, `current_a_surf` etc. → move into viewmodel dataclass.

## Next Steps
1. Define a `SubstrateSpotState` dataclass capturing mutable dialog data.  
2. Implement a presenter skeleton with methods for each major user action (add spot, remove, change lattice, compute transform).  
3. Introduce a thin scene adapter that the dialog holds; presenter provides overlay DTOs.  
4. Gradually move logic out of the dialog, rewriting dialog methods to delegate to presenter/scene.  
5. Add unit tests covering presenter logic independent of Qt.  
6. Adjust existing GUI tests (or add new pytest-qt tests) to ensure view interacts with presenter correctly.
