import types

from lfa.gui.visualizers.real_space_state import RealSpaceVisualizerState


def _make_controller(**attrs):
    ctrl = types.SimpleNamespace(**attrs)
    return ctrl


def test_substrate_offset_defaults_and_updates():
    controller = _make_controller()
    state = RealSpaceVisualizerState(controller)

    assert state.get_substrate_offset() == (0.0, 0.0)
    assert state.set_substrate_offset((1, 2))
    assert controller.substrate_visual_offset_nm == (1.0, 2.0)
    # Same value should not trigger an update
    assert not state.set_substrate_offset((1.0, 2.0))


def test_substrate_offset_normalises_iterables():
    controller = _make_controller(substrate_visual_offset_nm=[3, -4])
    state = RealSpaceVisualizerState(controller)
    assert state.get_substrate_offset() == (3.0, -4.0)


def test_adsorbate_offset_defaults_created():
    controller = _make_controller()
    state = RealSpaceVisualizerState(controller)

    assert state.get_adsorbate_offset(0) == (0.0, 0.0)
    assert controller.adsorbate_visual_offsets_nm[0] == (0.0, 0.0)


def test_adsorbate_offset_updates_by_set_index():
    controller = _make_controller(adsorbate_visual_offsets_nm={})
    state = RealSpaceVisualizerState(controller)

    assert state.set_adsorbate_offset(1, (0.1, -0.2))
    assert controller.adsorbate_visual_offsets_nm[1] == (0.1, -0.2)
    assert not state.set_adsorbate_offset(1, (0.1, -0.2))


def test_adsorbate_offset_invalid_index_is_ignored():
    controller = _make_controller()
    state = RealSpaceVisualizerState(controller)
    assert not state.set_adsorbate_offset(None, (1, 1))
    assert state.get_adsorbate_offset(None) == (0.0, 0.0)
