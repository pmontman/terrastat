import datetime as dt

from terrastat.timeparse import canonical_frequency, frequency_of_period, period_to_date


def test_period_spellings():
    assert period_to_date("2015") == dt.date(2015, 1, 1)
    assert period_to_date("2015-S2") == dt.date(2015, 7, 1)
    assert period_to_date("2015S1") == dt.date(2015, 1, 1)
    assert period_to_date("2015-Q4") == dt.date(2015, 10, 1)
    assert period_to_date("2015Q2") == dt.date(2015, 4, 1)
    assert period_to_date("2015-02") == dt.date(2015, 2, 1)
    assert period_to_date("2015M02") == dt.date(2015, 2, 1)
    assert period_to_date("2015-W05") == dt.date(2015, 1, 26)
    assert period_to_date("2015-12-31") == dt.date(2015, 12, 31)
    assert period_to_date("2015M12D31") == dt.date(2015, 12, 31)
    assert period_to_date("LTAA") is None
    assert period_to_date("2015-13") is None


def test_frequency_of_period():
    assert frequency_of_period("2015") == "A"
    assert frequency_of_period("2015-Q1") == "Q"
    assert frequency_of_period("2015-01") == "M"
    assert frequency_of_period("2015W01") == "W"
    assert frequency_of_period("2015-01-02") == "D"
    assert frequency_of_period("junk") == "OTHER"


def test_canonical_frequency_fred():
    # every string FRED uses across all 845,518 series of its 331 releases
    assert canonical_frequency("Annual", "fred") == "A"
    assert canonical_frequency("Monthly", "fred") == "M"
    assert canonical_frequency("Quarterly", "fred") == "Q"
    assert canonical_frequency("Daily, 7-Day", "fred") == "D"
    assert canonical_frequency("Weekly, Ending Friday", "fred") == "W"
    assert canonical_frequency("Semiannual", "fred") == "S"
    assert canonical_frequency("5 Year", "fred") == "P"
    assert canonical_frequency("10 Year", "fred") == "P"
    assert canonical_frequency("Biweekly, Ending Wednesday", "fred") == "BW"
    assert canonical_frequency("Not Applicable", "fred") == "OTHER"


def test_canonical_frequency_sdmx():
    # the full Eurostat ESTAT:FREQ codelist (v3.9)
    assert canonical_frequency("P", "eurostat") == "P"
    assert canonical_frequency("A", "eurostat") == "A"
    assert canonical_frequency("S", "eurostat") == "S"
    assert canonical_frequency("Q", "eurostat") == "Q"
    assert canonical_frequency("M", "eurostat") == "M"
    assert canonical_frequency("W", "eurostat") == "W"
    assert canonical_frequency("B", "eurostat") == "B"
    assert canonical_frequency("D", "eurostat") == "D"
    assert canonical_frequency("H", "eurostat") == "H"  # Hourly, NOT semiannual
    assert canonical_frequency("I", "eurostat") == "I"
    assert canonical_frequency("NAP", "eurostat") == "OTHER"
    assert canonical_frequency("A3", "eurostat") == "A3"
    assert canonical_frequency("Q", "oecd") == "Q"
    assert canonical_frequency("m", "oecd") == "M"  # case insensitive
    assert canonical_frequency(None) == "OTHER"
    assert canonical_frequency("ZZZ", "eurostat") == "OTHER"


def test_sentinel_periods_are_not_dates():
    # Eurostat's env_wat_ltaa writes "9999" for a long-term average, not a year
    assert period_to_date("9999") is None
    assert frequency_of_period("9999") == "OTHER"
    assert period_to_date("LTAA") is None
    assert period_to_date("2015") == dt.date(2015, 1, 1)  # a real year still parses
    assert period_to_date("1209") == dt.date(1209, 1, 1)  # the Bank of England millennium data


def test_hourly_periods():
    assert period_to_date("2015-01-02T13:00:00") == dt.date(2015, 1, 2)
    assert frequency_of_period("2015-01-02T13:00:00") == "H"
    assert frequency_of_period("2015-01-02") == "D"
