from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mortar_calculator as mortar  # noqa: E402


TEAM_SCREENSHOT = PROJECT_ROOT / "206376~1.JPG"
SCOPE_SCREENSHOT = PROJECT_ROOT / "2067AB~1.JPG"


@pytest.fixture(scope="module")
def team_frame() -> np.ndarray:
    frame = cv2.imread(str(TEAM_SCREENSHOT))
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def scope_frame() -> np.ndarray:
    frame = cv2.imread(str(SCOPE_SCREENSHOT))
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def ocr_engine():
    return mortar.create_ocr_engine()


@pytest.fixture(scope="module")
def read_team_result(team_frame: np.ndarray, ocr_engine):
    layout = mortar.ScreenLayout.from_frame(team_frame)
    return mortar.read_team_mortar_coordinates(team_frame, layout, ocr_engine)


@pytest.fixture(scope="module")
def read_target_result(team_frame: np.ndarray, ocr_engine):
    layout = mortar.ScreenLayout.from_frame(team_frame)
    return mortar.read_target_coordinates(team_frame, layout, ocr_engine)


def test_parse_coordinate_texts() -> None:
    point = mortar.parse_coordinate_texts(["y 61.07", "X97,77"])
    assert point == mortar.MapPoint(x=97.77, y=61.07)


def test_parse_coordinate_texts_requires_both_axes() -> None:
    with pytest.raises(mortar.CalculatorError, match="Y coordinate"):
        mortar.parse_coordinate_texts(["x97.77"])


def test_distance_uses_100_m_grid_squares() -> None:
    mortar_position = mortar.MapPoint(x=10.0, y=20.0)
    target = mortar.MapPoint(x=13.0, y=24.0)
    assert mortar.calculate_distance_m(mortar_position, target) == pytest.approx(
        500.0
    )


def test_rounds_to_nearest_five_metres() -> None:
    assert mortar.round_to_nearest_five(412.4) == 410
    assert mortar.round_to_nearest_five(412.5) == 415
    assert mortar.round_to_nearest_five(417.5) == 420


def test_detects_mortar_scope_only(
    scope_frame: np.ndarray, team_frame: np.ndarray
) -> None:
    assert mortar.is_mortar_scope(scope_frame)
    assert not mortar.is_mortar_scope(team_frame)


def test_newest_team_row_is_selected() -> None:
    recognized = [
        mortar.OcrText("TEAM", 0.99, 10.0, 20.0),
        mortar.OcrText("x1.00, y2.00", 0.99, 60.0, 20.0),
        mortar.OcrText("TEAM", 0.99, 10.0, 70.0),
        mortar.OcrText("x98.59, y109.82", 0.99, 60.0, 70.0),
    ]

    point = mortar._find_latest_team_coordinate(recognized, row_tolerance=10.0)
    assert point == mortar.MapPoint(x=98.59, y=109.82)


def test_coordinate_without_team_label_is_rejected() -> None:
    recognized = [mortar.OcrText("x98.59, y109.82", 0.99, 60.0, 20.0)]
    with pytest.raises(mortar.CalculatorError, match="TEAM mortar coordinate"):
        mortar._find_latest_team_coordinate(recognized, row_tolerance=10.0)


def test_reads_team_mortar_coordinate(
    read_team_result,
) -> None:
    point, recognized_text = read_team_result

    assert point == mortar.MapPoint(x=98.59, y=109.82)
    assert "TEAM" in recognized_text
    assert any(text.startswith("x98.59") for text in recognized_text)


def test_reads_mouse_target_coordinate(
    read_target_result,
) -> None:
    target, recognized_text = read_target_result

    assert target == mortar.MapPoint(x=98.17, y=109.88)
    assert "y109.88" in recognized_text
    assert "x98.17" in recognized_text


def test_team_screenshot_calculates_42_m(
    read_team_result, read_target_result
) -> None:
    mortar_position, _ = read_team_result
    target, _ = read_target_result
    distance = mortar.calculate_distance_m(mortar_position, target)

    assert distance == pytest.approx(42.4264, abs=0.001)
    assert round(distance) == 42


def test_session_locks_first_team_coordinate_until_rearmed() -> None:
    session = mortar.MortarSession()
    first = mortar.MapPoint(98.59, 109.82)
    ignored = mortar.MapPoint(10.0, 20.0)

    assert session.save_mortar_if_armed(first)
    assert not session.save_mortar_if_armed(ignored)
    assert session.snapshot().mortar == first
    assert not session.snapshot().team_watch_armed

    session.rearm_team_watch()
    assert session.snapshot().mortar is None
    assert session.snapshot().team_watch_armed
    assert session.save_mortar_if_armed(ignored)
    assert session.snapshot().mortar == ignored


def test_f8_requires_an_automatically_saved_mortar_position() -> None:
    with pytest.raises(mortar.CalculatorError, match="No mortar position"):
        mortar.require_saved_mortar(None)
