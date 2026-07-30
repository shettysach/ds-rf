from input_node import _format_status
from shared.messages import RuntimeStatus, StatusState


def test_non_error_statuses_are_hidden() -> None:
    assert _format_status(RuntimeStatus("motion-gen", StatusState.READY)) is None
    assert _format_status(RuntimeStatus("motion-gen", StatusState.GENERATING)) is None
    assert _format_status(RuntimeStatus("sonic", StatusState.PLAYING)) is None
    assert _format_status(RuntimeStatus("sonic", StatusState.DONE)) is None


def test_actionable_statuses_remain_visible() -> None:
    assert (
        _format_status(RuntimeStatus("sonic", StatusState.ERROR, detail="bad motion"))
        == "[sonic] error (bad motion)"
    )
