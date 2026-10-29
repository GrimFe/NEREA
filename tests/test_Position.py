import pandas as pd
import pytest
from datetime import datetime, timedelta

from nerea.time_series import Position

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in range(7)],
        'value': [1, 1, 1, 2, 20, 20, 20]
    })
    return data

@pytest.fixture
def sample_data_2s_timebase():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i * 2) for i in range(7)],
        'value': [1, 1, 1, 2, 20, 20, 20]
    })
    return data

@pytest.fixture
def position(sample_data):
    return Position(sample_data, timebase=1.)

@pytest.fixture
def position_2s_timebase(sample_data_2s_timebase):
    return Position(sample_data_2s_timebase, timebase=2.)

def test_timebase(position, position_2s_timebase):
    assert position.timebase == 1
    assert position_2s_timebase.timebase == 2

def test_plateau(position, position_2s_timebase):
    data = position.data
    plateau = pd.DataFrame({
        0: [data["Time"][0], data["Time"][2], 1, 1, 3],
        1: [data["Time"][3], data["Time"][3], 2, 2, 1],
        2: [data["Time"][4], data["Time"][6], 20, 20, 3],
    }, index=["start", "end", "first value", "mean value","length"])
    pd.testing.assert_frame_equal(position.plateau(tol=0,
                                                   absolute_tolerance=True),
                                  plateau)
    plateau = pd.DataFrame({
        0: [data["Time"][0], data["Time"][2], 1, 1, 3],
        1: [data["Time"][4], data["Time"][6], 20, 20, 3],
    }, index=["start", "end", "first value", "mean value", "length"])
    pd.testing.assert_frame_equal(position.plateau(tol=0,
                                                   min_length=2,
                                                   absolute_tolerance=True),
                                  plateau)
    plateau = pd.DataFrame({
        0: [data["Time"][0], data["Time"][3], 1, 1.25, 4],
        1: [data["Time"][4], data["Time"][6], 20, 20., 3],
    }, index=["start", "end", "first value", "mean value", "length"])
    pd.testing.assert_frame_equal(position.plateau(tol=1,
                                                   absolute_tolerance=True),
                                  plateau)
    plateau = pd.DataFrame({
        0: [data["Time"][0], data["Time"][6], 1, 9.28571428571, 7],
        }, index=["start", "end", "first value", "mean value", "length"])
    pd.testing.assert_frame_equal(position.plateau(tol=1,
                                                   absolute_tolerance=False),
                                  plateau)
    plateau = pd.DataFrame({
        0: [data["Time"][0], data["Time"][3], 0., 0.875, 4],
        1: [data["Time"][4], data["Time"][4], 11., 11., 1],
        2: [data["Time"][5], data["Time"][6], 20., 20., 2],
        }, index=["start", "end", "first value", "mean value", "length"])
    pd.testing.assert_frame_equal(position.plateau(tol=2,
                                                   absolute_tolerance=True,
                                                   smooth=True,
                                                   smoothing_method='moving_average',
                                                   window=2,
                                                   renormalize=False),
                                  plateau)
    
    # test skip_left and skip_right
    data = position_2s_timebase.data
    plateau = pd.DataFrame({
        0: [data["Time"][1], data["Time"][1], 1, 1, 1],
        1: [data["Time"][5], data["Time"][5], 20, 20, 1],
    }, index=["start", "end", "first value", "mean value","length"])
    pd.testing.assert_frame_equal(position_2s_timebase.plateau(
                                                tol=0,
                                                absolute_tolerance=True,
                                                skip_left=2, # seconds = 1 bin
                                                skip_right=2),
                                  plateau)
    # np.floor() applied if fractional bins need to be skipped
    data = position_2s_timebase.data
    plateau = pd.DataFrame({
        0: [data["Time"][1], data["Time"][2], 1, 1, 2],
        1: [data["Time"][5], data["Time"][6], 20, 20, 2],
    }, index=["start", "end", "first value", "mean value","length"])
    pd.testing.assert_frame_equal(position_2s_timebase.plateau(
                                                tol=0,
                                                absolute_tolerance=True,
                                                skip_left=2, # seconds = 1 bin
                                                skip_right=1),
                                  plateau)


def test_from_plateau(position):
    data = position.data
    result = position.from_plateau(tol=0)

    # --- check plateau labels ---
    assert set(result.data["PLATEAU"]) == {0, 1, 2}

    grouped = result.data.groupby("PLATEAU")

    # --- plateau 0 ---
    p0 = grouped.get_group(0)
    assert p0["value"].eq(1).all()
    assert p0["Time"].min() == data["Time"][0]
    assert p0["Time"].max() == data["Time"][2]

    # --- plateau 1 (single point) ---
    p1 = grouped.get_group(1)
    assert p1["value"].eq(2).all()
    assert p1["Time"].min() == data["Time"][3]
    assert p1["Time"].max() == data["Time"][3]

    # --- plateau 2 ---
    p2 = grouped.get_group(2)
    assert p2["value"].eq(20).all()
    assert p2["Time"].min() == data["Time"][4]
    assert p2["Time"].max() == data["Time"][6]  # end excluded
