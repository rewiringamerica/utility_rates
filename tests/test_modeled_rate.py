"""Tests for Rate and RateSchedule classes."""

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday

from rates.modeled_rate import (
    FixedChargeOption,
    ModeledRate,
    ModeledRateWithWeekendHoliday,
)
from rates.rate_structure import (
    DaysApplicable,
    FlatRateSchedule,
    Rate,
    RateIndexSchedule,
    RateSchedule,
)


def expand_timestamps(df):
    """Replace a `timestamp` column with `month`, `day`, and `hour` columns."""
    df["month"] = df.timestamp.dt.month
    df["day"] = df.timestamp.dt.day
    df["hour"] = df.timestamp.dt.hour
    return df.drop("timestamp", axis=1)


@pytest.fixture
def sample_load_profile_df():
    """Create a sample load profile DataFrame with three days for testing."""
    timestamps = pd.date_range(start="2023-01-01", end="2023-01-03 23:00:00", freq="H")
    building_ids = [1, 2] * len(timestamps)
    electricity_kwh = np.random.uniform(0.5, 5, size=len(timestamps) * 2)
    load_profile_df = pd.DataFrame(
        {"timestamp": np.repeat(timestamps, 2), "building_id": building_ids, "electricity_kwh": electricity_kwh}
    )
    return expand_timestamps(load_profile_df)


@pytest.fixture
def sample_load_profile_with_upgrades_df():
    """Create a sample load profile DataFrame with two upgrades and three days for testing."""
    timestamps = pd.date_range(start="2023-01-01", end="2023-01-03 23:00:00", freq="H")
    length = len(timestamps)

    baseline = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "building_id": [1, 2] * length,
            "electricity_kwh": [0.5] * 2 * length,
            "upgrade_id": [0] * 2 * length,
        }
    )
    baseline = expand_timestamps(baseline)

    upgrade = baseline.copy()
    upgrade["upgrade_id"] = 5
    upgrade["electricity_kwh"] *= 2

    return pd.concat([baseline, upgrade])


@pytest.fixture
def annual_load_profile_df():
    """Create an annual load profile with constant 1 kWh consumption."""
    timestamps = pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="H")
    building_ids = [1, 2] * len(timestamps)
    electricity_kwh = np.ones(len(timestamps) * 2)  # All 1s
    load_profile_df = pd.DataFrame(
        {"timestamp": np.repeat(timestamps, 2), "building_id": building_ids, "electricity_kwh": electricity_kwh}
    )
    load_profile_df["month"] = load_profile_df.timestamp.dt.month
    load_profile_df["day"] = load_profile_df.timestamp.dt.day
    load_profile_df["hour"] = load_profile_df.timestamp.dt.hour

    return load_profile_df.drop("timestamp", axis=1)


@pytest.fixture
def flat_rate_model(sample_load_profile_df):
    """Create a ModeledRate instance with flat rate."""
    return ModeledRate(
        load_profile_df=sample_load_profile_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )


@pytest.fixture
def tou_rate_model(sample_load_profile_df):
    """Create a ModeledRate instance with time-of-use rates."""
    # Create a TOU schedule where hours 12-17 (noon-5pm) are peak
    rate_schedule = np.zeros((12, 24), dtype=int)
    rate_schedule[:, 12:18] = 1  # Peak hours is 1, off peak is 0
    return ModeledRate(
        load_profile_df=sample_load_profile_df,
        rate_schedule=RateIndexSchedule(np.array([0.10, 0.20]), rate_schedule, 5.0),
    )


@pytest.fixture
def seasonal_tou_rate_model(annual_load_profile_df):
    """Create a ModeledRate instance with seasonal TOU rates (winter/summer on/off-peak)."""
    new_rate_schedule = RateSchedule(
        rates=[
            Rate(
                volumetric_rate=0.10,
                name="Summer off peak",
            ),
            Rate(
                volumetric_rate=0.20,
                tou_start_hour=12,
                tou_end_hour=18,
                name="Summer on peak",
            ),
            Rate(
                volumetric_rate=0.08,
                seasonal_start_month=10,
                seasonal_end_month=6,
                name="Winter off peak",
            ),
            Rate(
                volumetric_rate=0.18,
                seasonal_start_month=10,
                seasonal_end_month=6,
                tou_start_hour=12,
                tou_end_hour=18,
                name="Winter on peak",
            ),
        ],
        monthly_fixed_charge=5.0,
    )

    return ModeledRate(
        load_profile_df=annual_load_profile_df,
        rate_schedule=new_rate_schedule,
        name="TOU and seasonal rate",
        fuel="electricity",
    )


@pytest.fixture
def weekend_holiday_rate_model(sample_load_profile_df):
    """Create a ModeledRateWithWeekendHoliday instance with different weekend/holiday rates."""

    # Make Jan 2nd a holiday since Jan 1st is a Sunday
    class TestHolidays(AbstractHolidayCalendar):
        rules = [Holiday("Test Holiday", month=1, day=2)]

    # Weekday rate schedule
    weekday_rate_matrix = np.full((12, 24), 0.10)
    weekday_rate_matrix[:, 12:18] = 0.20

    # Weekend/holiday rate schedule
    weekend_rate_matrix = np.full((12, 24), 0.05)
    weekend_rate_matrix[:, 18:23] = 0.25

    rate_schedule = RateSchedule(
        rates=[
            Rate(
                volumetric_rate=0.10,
                days_applicable=DaysApplicable.WEEKDAYS,
            ),
            Rate(
                volumetric_rate=0.20,
                tou_start_hour=12,
                tou_end_hour=18,
                days_applicable=DaysApplicable.WEEKDAYS,
            ),
            Rate(
                volumetric_rate=0.05,
                days_applicable=DaysApplicable.WEEKENDS_HOLIDAYS,
            ),
            Rate(
                volumetric_rate=0.25,
                tou_start_hour=18,
                tou_end_hour=23,
                days_applicable=DaysApplicable.WEEKENDS_HOLIDAYS,
            ),
        ],
        monthly_fixed_charge=5.0,
    )

    model = ModeledRateWithWeekendHoliday(
        load_profile_df=sample_load_profile_df,
        rate_schedule=rate_schedule,
        holiday_calendar=TestHolidays(),
        data_year=2023,
    )

    return model


# -- Test ModeledRate -- #
def test_initialization_with_flat_rate(flat_rate_model):
    """Test initialization with flat rate."""
    assert np.array_equal(flat_rate_model.building_ids, np.array([1, 2]))
    assert flat_rate_model.monthly_fixed_charge == 10.0
    assert np.all(flat_rate_model.rate_matrix == 0.15)
    assert len(flat_rate_model.building_ids) == 2
    assert flat_rate_model.load_profile_matrix.shape == (2, 12, 24, 31)


def test_initialization_with_tou_rate(tou_rate_model):
    """Test initialization with time-of-use rates."""
    assert tou_rate_model.monthly_fixed_charge == 5.0
    assert tou_rate_model.rate_matrix.shape == (12, 24)
    # Check that peak hours have the correct rate
    assert np.all(tou_rate_model.rate_matrix[:, 12:18] == 0.20)
    assert np.all(tou_rate_model.rate_matrix[:, :12] == 0.10)
    assert np.all(tou_rate_model.rate_matrix[:, 18:] == 0.10)


def test_initialization_with_seasonal_tou_rates(seasonal_tou_rate_model):
    """Test seasonal TOU rate initialization and rate matrix."""
    model = seasonal_tou_rate_model

    # Basic properties
    assert model.monthly_fixed_charge == 5.0
    assert model.name == "TOU and seasonal rate"
    assert model.fuel == "electricity"

    # Summer months (June-Sept) - index 5-8
    for month in [5, 6, 7, 8]:
        # Summer on-peak (12-18)
        assert np.all(model.rate_matrix[month, 12:18] == 0.20)
        # Summer off-peak (all other hours)
        assert np.all(model.rate_matrix[month, :12] == 0.10)
        assert np.all(model.rate_matrix[month, 18:] == 0.10)

    # Winter months (rest of year)
    winter_months = [m for m in range(12) if m not in [5, 6, 7, 8]]
    for month in winter_months:
        # Winter on-peak (12-18)
        assert np.all(model.rate_matrix[month, 12:18] == 0.18)
        # Winter off-peak (all other hours)
        assert np.all(model.rate_matrix[month, :12] == 0.08)
        assert np.all(model.rate_matrix[month, 18:] == 0.08)


def test_initialization_with_upgrades(sample_load_profile_with_upgrades_df):
    """Test initializing with a load profile that contains multiple upgrades."""
    models = ModeledRate.create_modeled_rate_per_upgrade(
        load_profile_upgrades_df=sample_load_profile_with_upgrades_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )

    assert set(models.keys()) == {"0", "5"}

    for upgrade_id, model in models.items():
        assert np.array_equal(model.building_ids, np.array([1, 2]))
        assert model.monthly_fixed_charge == 10.0
        assert np.all(model.rate_matrix == 0.15)
        # Check that each upgrade has the correct load
        load_kwh = 0.5 if upgrade_id == "0" else 1
        assert np.all(model.load_profile_matrix[:, 0, 0:3, 0] == load_kwh)


def test_initialization_from_schedule_matrix(sample_load_profile_df):
    """Test initialization from a schedule matrix and list of volumetric rates."""
    # Create a TOU schedule where hours 12-17 (noon-5pm) are peak
    rate_schedule = np.zeros((12, 24), dtype=int)
    rate_schedule[:, 12:18] = 1  # Peak hours is 1, off peak is 0

    tou_rate_model = ModeledRate(
        load_profile_df=sample_load_profile_df,
        rate_schedule=RateIndexSchedule(
            volumetric_rates=np.array([0.10, 0.20]),  # Off-peak and peak rates
            rate_schedule_matrix=rate_schedule,
            monthly_fixed_charge=5.0,
        ),
    )

    assert tou_rate_model.monthly_fixed_charge == 5.0
    assert tou_rate_model.rate_matrix.shape == (12, 24)
    # Check that peak hours have the correct rate
    assert np.all(tou_rate_model.rate_matrix[:, 12:18] == 0.20)
    assert np.all(tou_rate_model.rate_matrix[:, :12] == 0.10)
    assert np.all(tou_rate_model.rate_matrix[:, 18:] == 0.10)


def test_build_load_profile_matrix(flat_rate_model, sample_load_profile_df):
    """Test building the load profile matrix."""
    matrix = flat_rate_model.load_profile_matrix
    assert matrix.shape == (2, 12, 24, 31)

    # Check that values are correctly placed in the matrix
    for _, row in sample_load_profile_df.iterrows():
        month = int(row["month"] - 1)
        hour = int(row["hour"])
        day = int(row["day"] - 1)
        building_idx = 0 if row["building_id"] == 1 else 1
        assert matrix[building_idx, month, hour, day] == row["electricity_kwh"]

    # Check that non-existent days are filled with 0
    assert np.all(matrix[:, :, :, 30] == 0)  # Day 31 (index 30) shouldn't exist in Jan


def test_bills_by_building_month_hour_property_flat_rate(flat_rate_model):
    """Test the bills_by_building_month_hour property for a flat rate."""
    bills = flat_rate_model.bills_by_building_month_hour
    assert bills.shape == (2, 12, 24)

    # Check that the property is cached
    assert flat_rate_model._bills_by_building_month_hour is not None

    # Check calculation is correct (rate * consumption summed over days)
    expected = flat_rate_model.load_profile_matrix.sum(axis=3) * 0.15
    assert np.allclose(bills, expected)


def test_bills_by_building_month_hour_property_tou_seasonal_rate(seasonal_tou_rate_model):
    """Test the bills_by_building_month_hour property for a seasonal tou rate."""
    n_days_jan = 31
    n_days_june = 30
    bills_by_building_month_hour = seasonal_tou_rate_model.bills_by_building_month_hour
    # for each period, and a given building, check that each total cost summed over all days
    # in a given (month, hour) matches expected rates times number of days in the month
    assert bills_by_building_month_hour[0, 0, 0] == 0.08 * n_days_jan  # Winter off-peak
    assert bills_by_building_month_hour[0, 0, 13] == 0.18 * n_days_jan  # Winter on-peak
    assert bills_by_building_month_hour[0, 5, 0] == 0.10 * n_days_june  # Summer off-peak
    assert bills_by_building_month_hour[0, 5, 13] == 0.20 * n_days_june  # Summer on-peak
    # check the same result for both buildings since both have identical 1kwh per day usage
    assert bills_by_building_month_hour[0, 0, 0] == bills_by_building_month_hour[1, 0, 0]


def test_calculate_monthly_bills():
    """Test calculating monthly bills with different fixed charge options."""
    # Create a sample load profile with one building having zero consumption
    timestamps = pd.date_range(start="2023-01-01", end="2023-01-03 23:00:00", freq="H")
    building_ids = [1] * len(timestamps) + [2] * len(timestamps)

    # Building 1 has zero consumption, Building 2 has normal consumption
    electricity_kwh = np.concatenate(
        [np.zeros(len(timestamps)), np.random.uniform(0.5, 5, size=len(timestamps))]  # Building 1  # Building 2
    )

    load_profile_df = pd.DataFrame(
        {"timestamp": np.repeat(timestamps, 2), "building_id": building_ids, "electricity_kwh": electricity_kwh}
    )
    load_profile_df["month"] = load_profile_df.timestamp.dt.month
    load_profile_df["day"] = load_profile_df.timestamp.dt.day
    load_profile_df["hour"] = load_profile_df.timestamp.dt.hour

    # Initialize the rate model
    rate_model = ModeledRate(
        load_profile_df=load_profile_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )

    # Calculate expected volumetric costs (sum across days)
    volumetric_cost = rate_model.bills_by_building_month_hour.sum(axis=2)

    # Test "never" option
    monthly_bills_never = rate_model.calculate_monthly_bills(fixed_charge_option=FixedChargeOption.NEVER)
    assert np.allclose(monthly_bills_never, volumetric_cost)

    # Test "always" option
    monthly_bills_always = rate_model.calculate_monthly_bills(fixed_charge_option=FixedChargeOption.ALWAYS)
    expected_always = volumetric_cost + 10.0
    assert np.allclose(monthly_bills_always, expected_always)

    # Test "if_consumption" option
    monthly_bills_if_consumption = rate_model.calculate_monthly_bills(
        fixed_charge_option=FixedChargeOption.IF_CONSUMPTION
    )

    # Building 1 should have no fixed charge (zero consumption)
    assert np.allclose(monthly_bills_if_consumption[0], volumetric_cost[0])

    # Building 2 should have fixed charge (has consumption)
    assert np.allclose(monthly_bills_if_consumption[1], volumetric_cost[1] + 10.0)


def test_calculate_annual_bills(flat_rate_model):
    """Test calculating annual bills."""
    annual_bills = flat_rate_model.calculate_annual_bills()
    assert annual_bills.shape == (2,)

    # Check that fixed charge is included (12 months * $10)
    expected = flat_rate_model.bills_by_building_month_hour.sum(axis=(1, 2)) + 10.0 * 12
    assert np.allclose(annual_bills, expected)

    # Test without fixed charge
    annual_bills_no_fixed = flat_rate_model.calculate_annual_bills(fixed_charge_option=FixedChargeOption.NEVER)
    expected_no_fixed = flat_rate_model.bills_by_building_month_hour.sum(axis=(1, 2))
    assert np.allclose(annual_bills_no_fixed, expected_no_fixed)


def test_tou_rate_calculations(tou_rate_model, sample_load_profile_df):
    """Test calculations with time-of-use rates."""
    # Get the load for peak hours (12-17) in the sample data
    peak_hours = sample_load_profile_df[(sample_load_profile_df.hour >= 12) & (sample_load_profile_df.hour < 18)]
    peak_load = peak_hours.groupby("building_id")["electricity_kwh"].sum().values

    # Get the load for off-peak hours
    off_peak_hours = sample_load_profile_df[(sample_load_profile_df.hour < 12) | (sample_load_profile_df.hour >= 18)]
    off_peak_load = off_peak_hours.groupby("building_id")["electricity_kwh"].sum().values

    # Calculate expected annual cost
    expected_cost = (peak_load * 0.20) + (off_peak_load * 0.10) + (12 * 5.0)

    # Compare with calculated annual bills
    assert np.allclose(tou_rate_model.calculate_annual_bills(), expected_cost)


def test_to_long_format(flat_rate_model):
    """Test converting rate calculation results to long format."""
    upgrade_id = 0
    base_columns = {"building_id", "energy", "cost", "upgrade_id", "fuel", "rate_name"}

    # Test annual granularity
    annual_df = flat_rate_model.to_long_format(upgrade_id, granularity="annual")

    # Check columns and shape
    assert set(annual_df.columns) == base_columns
    assert annual_df.shape[0] == 2  # One row per building

    # Check values
    assert np.allclose(annual_df["energy"], flat_rate_model.load_profile_matrix.sum(axis=(1, 2, 3)))
    assert np.allclose(annual_df["cost"], flat_rate_model.calculate_annual_bills())
    assert all(annual_df["upgrade_id"] == upgrade_id)
    assert all(annual_df["fuel"] == flat_rate_model.fuel)
    assert all(annual_df["rate_name"] == flat_rate_model.name)

    # Test monthly granularity
    monthly_df = flat_rate_model.to_long_format(upgrade_id, granularity="monthly")

    # Check columns and shape
    assert set(monthly_df.columns) == base_columns.union({"month"})
    assert monthly_df.shape[0] == 2 * 12  # One row per building per month

    # Check month values
    assert sorted(monthly_df["month"].unique()) == list(range(1, 13))
    assert all(monthly_df.groupby("building_id")["month"].nunique() == 12)

    # Test hourly granularity
    hourly_df = flat_rate_model.to_long_format(upgrade_id, granularity="hourly")

    # Check columns and shape
    assert set(hourly_df.columns) == base_columns.union({"hour"})
    assert hourly_df.shape[0] == 2 * 12 * 24  # One row per building per hour per month

    # Check hour values
    assert sorted(hourly_df["hour"].unique()) == list(range(24))

    # Test monthly_hourly granularity
    monthly_hourly_df = flat_rate_model.to_long_format(upgrade_id, granularity="monthly_hourly")

    # Check columns and shape
    assert set(monthly_hourly_df.columns) == base_columns.union(
        {"hour", "month", "cost_daily_mean", "energy_daily_mean", "days_in_month"}
    )
    assert monthly_hourly_df.shape[0] == 2 * 12 * 24  # One row per building per hour per month\

    # Check month and hour combinations
    assert all(monthly_hourly_df.groupby(["building_id", "month"])["hour"].nunique() == 24)

    # Test invalid granularity
    with pytest.raises(ValueError):
        flat_rate_model.to_long_format(upgrade_id, granularity="invalid")


def test_apply_baseline_credit():
    """Test apply baseline credit."""
    # Create test data for 1 building, for 1 yeae
    timestamps = pd.date_range(start="2023-01-01", periods=8760, freq="H")

    # Set simple consumption pattern: 6kWh at noon every day, 0 kWh otherwise
    electricity_kwh = np.zeros(len(timestamps))
    noon_usage = 6.0
    electricity_kwh[timestamps.hour == 12] = noon_usage

    load_profile_df = pd.DataFrame(
        {"timestamp": timestamps, "building_id": np.full(len(timestamps), 1), "electricity_kwh": electricity_kwh}
    )
    load_profile_df["month"] = load_profile_df.timestamp.dt.month
    load_profile_df["day"] = load_profile_df.timestamp.dt.day
    load_profile_df["hour"] = load_profile_df.timestamp.dt.hour

    # Initialize model
    volumetric_rate = 0.2  # $0.2/kWh
    model = ModeledRate(
        load_profile_df=load_profile_df,
        rate_schedule=FlatRateSchedule(volumetric_rate, 0),
    )

    # Calculate bills before baseline credit is applied
    pre_credit_bills = model.calculate_annual_bills()[0]

    # pply baseline credit (5kWh/day allowance, $0.10/hour credit)
    baseline_allowance = 5.0  # kWh/day
    baseline_credit = 0.10  # $/hour
    model.apply_baseline_credit(baseline_credit, baseline_allowance)

    # Calculate bills after  baseline credit is applied
    post_credit_bills = model.calculate_annual_bills()[0]

    # Verify
    annual_baseline_credit = 365 * baseline_allowance * baseline_credit
    assert np.isclose(post_credit_bills, pre_credit_bills - annual_baseline_credit, rtol=1e-4)


# -- Test ModeledRateWithWeekendHoliday -- #
def test_weekend_holiday_initialization(weekend_holiday_rate_model):
    """Test initialization of ModeledRateWithWeekendHoliday."""
    assert weekend_holiday_rate_model.monthly_fixed_charge == 5.0

    # Check weekday rate matrix
    assert np.all(weekend_holiday_rate_model.rate_matrix[:, 12:18] == 0.20)  # on peak
    assert np.all(weekend_holiday_rate_model.rate_matrix[:, :12] == 0.10)  # off peak
    assert np.all(weekend_holiday_rate_model.rate_matrix[:, 18:] == 0.10)  # off peak

    # Check weekend/holiday rate matrix
    assert np.all(weekend_holiday_rate_model.rate_matrix_weekend_holiday[:, 18:23] == 0.25)  # on peak
    assert np.all(weekend_holiday_rate_model.rate_matrix_weekend_holiday[:, :18] == 0.05)  # off peak
    assert np.all(weekend_holiday_rate_model.rate_matrix_weekend_holiday[:, 23:] == 0.05)  # off peak


def test_weekend_holiday_initialization_with_upgrades(sample_load_profile_with_upgrades_df, holiday_rate_schedule):
    """Test initializing with a load profile that contains multiple upgrades."""
    models = ModeledRateWithWeekendHoliday.create_modeled_rate_per_upgrade(
        load_profile_upgrades_df=sample_load_profile_with_upgrades_df,
        rate_schedule=holiday_rate_schedule,
        name="test rate with upgrade",
        fuel="electricity",
    )

    assert set(models.keys()) == {"0", "5"}

    for upgrade_id, model in models.items():
        assert model.monthly_fixed_charge == 5.25
        # assert np.array_equal(model.volumetric_rates, np.array([0.38811, 0.41646, 0.43573, 0.51917]))
        # assert np.array_equal(model.volumetric_rates_weekend_holiday, np.array([0.38811, 0, 0, 0, 0.2]))

        assert model.rate_matrix.shape == (12, 24)
        assert set(np.unique(model.rate_matrix)) == {0.38811, 0.41646, 0.43573, 0.51917}
        assert np.all(model.rate_matrix_weekend_holiday == 0.2)

        assert model.name == "test rate with upgrade"
        assert model.fuel == "electricity"
        # Check that each upgrade has the correct load
        load_kwh = 0.5 if upgrade_id == "0" else 1
        assert np.all(model.load_profile_matrix[:, 0, 0:3, 0] == load_kwh)


def test_weekend_holiday_mask(weekend_holiday_rate_model):
    """Test the weekend/holiday mask construction."""
    mask = weekend_holiday_rate_model.weekend_holiday_mask
    assert mask.shape == (12, 31)

    # In our test data (Jan 1-3, 2023):
    assert mask[0, 0] == True  # Jan 1 (weekend)
    assert mask[0, 1] == True  # Jan 2 (holiday)
    # All other days should be False
    assert np.all(mask[0, 2:] == False)
    assert np.all(mask[1:, :] == False)


def test_load_profile_splitting(weekend_holiday_rate_model):
    """Test that load profiles are correctly split between weekday and weekend/holiday."""
    # Total load should equal sum of weekday and weekend/holiday loads
    total_load = weekend_holiday_rate_model.load_profile_matrix.sum()
    weekday_load = weekend_holiday_rate_model.load_profile_matrix_weekday_non_holiday.sum()
    weekend_holiday_load = weekend_holiday_rate_model.load_profile_matrix_weekend_holiday.sum()

    assert np.isclose(total_load, weekday_load + weekend_holiday_load)

    # 2 days of weekend/holiday load (Jan 1 = weekend, Jan 2 = holiday)
    assert weekend_holiday_load > 0
    # One weekday in our period so this load should be less
    assert weekend_holiday_load > weekday_load


def test_weekend_holiday_bill_calculation(weekend_holiday_rate_model, sample_load_profile_df):
    """Test bill calculations with weekday and weekend/holiday rates."""
    # Get the dates for weekends/holidays and weekends
    weekend_holiday_hours = sample_load_profile_df[
        ((sample_load_profile_df.month == 1) & (sample_load_profile_df.day == 1))  # 2023-01-01 is a Sunday
        | ((sample_load_profile_df.month == 1) & (sample_load_profile_df.day == 2))  # 2023-01-02 is a Monday holiday
    ]
    weekday_hours = sample_load_profile_df[
        (sample_load_profile_df.month == 1) & (sample_load_profile_df.day == 3)  # 2023-01-03 is a weekday non-holiday
    ]

    # Get hours for each period
    weekend_peak_hours = weekend_holiday_hours[(weekend_holiday_hours.hour >= 18) & (weekend_holiday_hours.hour < 23)]
    weekend_off_peak_hours = weekend_holiday_hours[
        (weekend_holiday_hours.hour < 18) | (weekend_holiday_hours.hour >= 23)
    ]
    weekday_peak_hours = weekday_hours[(weekday_hours.hour >= 12) & (weekday_hours.hour < 18)]
    weekday_off_peak_hours = weekday_hours[(weekday_hours.hour < 12) | (weekday_hours.hour >= 18)]

    # Get loads for each rate period
    weekend_peak_load = weekend_peak_hours.groupby("building_id")["electricity_kwh"].sum().values
    weekend_off_peak_load = weekend_off_peak_hours.groupby("building_id")["electricity_kwh"].sum().values
    weekday_peak_load = weekday_peak_hours.groupby("building_id")["electricity_kwh"].sum().values
    weekday_off_peak_load = weekday_off_peak_hours.groupby("building_id")["electricity_kwh"].sum().values

    # Calculate expected cost components
    expected_weekend_costs = weekend_peak_load * 0.25 + weekend_off_peak_load * 0.05
    expected_weekday_costs = weekday_peak_load * 0.2 + weekday_off_peak_load * 0.10

    # Compare with calculated and expected
    assert np.allclose(
        weekend_holiday_rate_model.calculate_annual_bills(fixed_charge_option=FixedChargeOption.NEVER),
        expected_weekend_costs + expected_weekday_costs,
    ), "Total bills don't match expected"


def test_same_rate_weekend_weekday_flat_rate(sample_load_profile_df):
    """Test that same rate/schedule for weekend and weekday gives same result as superclass."""
    # Create a regular ModeledRate instance
    regular_rate = ModeledRate(
        load_profile_df=sample_load_profile_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )

    # Create a ModeledRateWithWeekendHoliday with identical rates/schedules
    weekend_holiday_rate = ModeledRateWithWeekendHoliday(
        load_profile_df=sample_load_profile_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )

    # Monthly bills should be identical
    assert np.allclose(regular_rate.calculate_monthly_bills(), weekend_holiday_rate.calculate_monthly_bills())


def test_same_rate_weekend_weekday(sample_load_profile_df):
    """Test that same rate/schedule for weekend and weekday gives same result as superclass."""
    # TODO: Don't use a flat rate in this test

    # Create a regular ModeledRate instance
    regular_rate = ModeledRate(
        load_profile_df=sample_load_profile_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )

    # Create a ModeledRateWithWeekendHoliday with identical rates/schedules
    weekend_holiday_rate = ModeledRateWithWeekendHoliday(
        load_profile_df=sample_load_profile_df,
        rate_schedule=FlatRateSchedule(0.15, 10.0),
    )

    # Monthly bills should be identical
    assert np.allclose(regular_rate.calculate_monthly_bills(), weekend_holiday_rate.calculate_monthly_bills())


def test_different_rate_weekend_weekday(sample_load_profile_df, holiday_rate_schedule):
    """Test that different rate/schedule for weekend and weekday gives different result as superclass."""
    # Create a regular ModeledRate instance
    regular_rate = ModeledRate(
        load_profile_df=sample_load_profile_df,
        rate_schedule=FlatRateSchedule(0.5, 5.25),
    )

    # Create a ModeledRateWithWeekendHoliday with different weekend rate
    weekend_holiday_rate = ModeledRateWithWeekendHoliday(
        load_profile_df=sample_load_profile_df,
        rate_schedule=holiday_rate_schedule,
        name="test name",
    )

    # Monthly bills should be different
    assert not np.allclose(regular_rate.calculate_monthly_bills(), weekend_holiday_rate.calculate_monthly_bills())

    # The weekend/holiday version should be cheaper because weekend rate is lower
    assert (weekend_holiday_rate.calculate_annual_bills() < regular_rate.calculate_annual_bills()).all()
