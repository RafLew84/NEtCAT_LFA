.. _migration-notes:

========================================
Migration Notes for the 2025 Refactor
========================================

This page summarises the breaking and notable behavioural changes introduced
while completing Milestones 1–6 of the 2025 refactor.  It is intended for users
who maintain custom scripts, downstream plugins, or legacy sessions saved with
earlier versions of LFA.

Session File Format
===================

The session payload schema now reports ``format_version = "2.0"``.  Loading an
older project automatically runs the migration chain defined in
``lfa.logic.session_migrations``:

``1.0 → 1.1``
   Original image metadata is promoted into dedicated lists:

   - ``history_data.original_images`` stores one record per root scan
     (``display_name``, ``image_id``, ``extra_metadata``).
   - ``history_data.original_order`` keeps the tab order used by the GUI.

``1.1 → 2.0``
   Controller state gains richer geometry information and consistent naming:

   - ``domain_wall_analysis_results`` is renamed to
     ``superstructure_periodicity_results``.
   - Visual layer offsets are normalised to tuples of floats:
     ``substrate_visual_offset_nm`` and ``adsorbate_visual_offsets_nm`` map
     integer indices to ``(dx, dy)`` values (missing or invalid input falls
     back to ``(0.0, 0.0)``).
   - Substrate and adsorbate spot correspondences are stored as dictionaries
     with ``{"raw": (x, y), "transformed": (x, y)}``, replacing the informal
     tuple-of-tuples format.
   - Pixel calibration uncertainty ``pixel_calibration_sigma_nm`` and
     covariance-backed uncertainties for lattice metrics are persisted.
   - Adsorbate visual offsets and spot-pair indexes are coerced to integers so
     they survive JSON round-trips.

Sessions written before version 2.0 continue to load; the migration normalises
the payload before ``SessionState.from_payload`` is invoked.  When distributing
session files to collaborators, recommend upgrading to the latest LFA first so
everyone works with the same schema.

History Events and Controller Signals
=====================================

``HistoryManager`` now emits structured events instead of the monolithic
``current_node_changed`` signal:

``active_node_changed``
   Carries an :class:`ActiveNodeChangedEvent` with the node ID, resolved
   :class:`~lfa.core.history.HistoryNode` instance, and a ``reason`` string
   (``"selection"``, ``"cleared"``, ``"forced"``, etc.).  This enables consumers
   to react differently to user-driven and programmatic updates.

``original_image_added`` / ``original_image_removed``
   Provide :class:`OriginalImageEvent` payloads whenever root scans enter or
   leave the tree, allowing UI components to refresh labels without re-reading
   the entire history.

Legacy slots connected to ``current_node_changed`` continue to receive the node
object, but new integrations should switch to ``active_node_changed`` to gain
access to the structured metadata.  The dedicated ``HistoryViewHandler`` class
used by the main window illustrates the recommended pattern: connect once to
``active_node_changed`` and derive all enable/disable logic from the event.

The :class:`lfa.logic.app_controller.AppController` has likewise slimmed down.
Workflows that previously reached directly into controller internals should now
use the injected services under ``lfa.logic.services``:

``HistoryOrchestrator``
   Encapsulates mutations to the history tree, publishes events, and coordinates
   the ``HistoryManager``.

``SpotSetService``
   Owns substrate/adsorbate selections, affine fitting, and uncertainty
   propagation for both raw and corrected spot sets.

``AnalysisExecutor``
   Runs FFT calculations, superstructure periodicity, Patterson analysis, and
   other CPU-heavy routines.

``SessionService``
   Loads/saves sessions and applies migrations; custom automation scripts should
   prefer its API over calling :class:`SessionSerializer` directly.

Dialog and UI Changes
=====================

Preprocessing
-------------

All preprocessing dialogs now inherit from
:class:`lfa.gui.dialogs.preprocessing.BasePreprocessingDialog`, which provides
common ROI handling, live preview toggles, and validation.  The concrete dialog
classes live under ``lfa.gui.dialogs.preprocessing`` (``blur``, ``denoising``,
``leveling``, ``median`` modules).  The legacy import path
``lfa.gui.dialogs.preprocessing_dialogs`` remains as a thin wrapper that issues
``DeprecationWarning``; update imports to the new package to silence the notice.

Superstructure Analysis
-----------------------

The former “Domain Walls Analysis” dialog has been renamed to
``superstructure_periodicity_dialog.SuperstructurePeriodicityDialog`` along with
corresponding menu entries (“Analyse Superstructure Periodicity”).  Internal
keys, controller attributes, and exported report fields follow the new naming,
so downstream tooling should likewise rename any references.

Real-Space Visualiser
---------------------

The real-space/FFT viewer now displays lattice vector uncertainties and stores
manual layer offsets in the session file.  Button enablement is driven by
``UIStateBinder`` (instead of ad-hoc checks inside the main window), meaning
custom widgets that toggle analyser actions should reuse the binder or emit the
same state-change signals.

Upgrade Checklist
=================

#. Update Python imports to reference ``lfa.gui.dialogs.preprocessing`` and the
   new service modules under ``lfa.logic.services``.
#. When listening for history changes, migrate to ``active_node_changed`` and
   the dataclass payloads.
#. Regenerate or resave any automated session fixtures so they include the
   ``format_version = "2.0"`` schema additions (pixel calibration sigma,
   normalised offsets, renamed superstructure results).
#. Review custom reports/exports to ensure they read the uncertainty fields
   introduced for rotation, stretch, and lattice metrics.
#. If you run Sphinx locally, add the new ``migration_notes`` page to your
   documentation build (see :ref:`migration-notes` in the main index).

Following these steps will keep bespoke tooling aligned with the refactored LFA
codebase and minimise surprises when sharing project files between versions.
