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


@pytest.fixture(scope="module")
def team_frame() -> np.ndarray:
    frame = cv2.imread(str(TEAM_SCREENSHOT))
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def ocr_engine():
    return mortar.create_ocr_engine()


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
    team_frame: np.ndarray, ocr_engine
) -> None:
    layout = mortar.ScreenLayout.from_frame(team_frame)
    point, recognized_text = mortar.read_team_mortar_coordinates(
        team_frame, layout, ocr_engine
    )

    assert point == mortar.MapPoint(x=98.59, y=109.82)
    assert "TEAM" in recognized_text
    assert any(text.startswith("x98.59") for text in recognized_text)


def test_reads_mouse_target_coordinate(
    team_frame: np.ndarray, ocr_engine
) -> None:
    layout = mortar.ScreenLayout.from_frame(team_frame)
    target, recognized_text = mortar.read_target_coordinates(
        team_frame, layout, ocr_engine
    )

    assert target == mortar.MapPoint(x=98.17, y=109.88)
    assert "y109.88" in recognized_text
    assert "x98.17" in recognized_text


def test_team_screenshot_calculates_42_m(
    team_frame: np.ndarray, ocr_engine
) -> None:
    layout = mortar.ScreenLayout.from_frame(team_frame)
    mortar_position, _ = mortar.read_team_mortar_coordinates(
        team_frame, layout, ocr_engine
    )
    result = mortar.analyze_target_frame(team_frame, mortar_position, ocr_engine)

    assert result.mortar == mortar.MapPoint(x=98.59, y=109.82)
    assert result.target == mortar.MapPoint(x=98.17, y=109.88)
    assert result.distance_m == pytest.approx(42.4264, abs=0.001)
    assert result.rounded_distance_m == 42


def test_f8_requires_a_saved_f7_mortar_position() -> None:
    with pytest.raises(mortar.CalculatorError, match="press F7 first"):
        mortar.require_saved_mortar(None)
