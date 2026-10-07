import logging

import pandas as pd

from src.data_prep.loader import EDUCATIONAL_WARNING, ensure_datetime_column, load_credit_data
from src.data_prep.out_of_time_split import out_of_time_split


def test_loader_warns_and_returns_copy(caplog):
    frame = pd.DataFrame({"target": [0, 1]})
    with caplog.at_level(logging.WARNING):
        result = load_credit_data(frame=frame)
    assert EDUCATIONAL_WARNING in caplog.text
    assert result.equals(frame)


def test_oot_split_is_strictly_temporal():
    frame = ensure_datetime_column(pd.DataFrame({"target": [0, 1, 0, 1, 0]}))
    split = out_of_time_split(frame, test_fraction=0.4)
    assert split.train["application_date"].max() < split.test["application_date"].min()