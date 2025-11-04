"""Shared fixtures for testing rate schedules."""

import pytest

from rates.rate_structure import DaysApplicable, Rate, RateSchedule


@pytest.fixture
def simple_rate_schedule():
    """Create a simple rate schedule to test with."""
    base_rate = Rate(volumetric_rate=0.2, name="non seasonal base")
    tou_rate = Rate(
        volumetric_rate=0.4,
        tou_start_hour=12,
        tou_end_hour=20,
        seasonal_start_month=6,
        seasonal_end_month=9,
        name="non seasonal TOU",
    )
    fixed_charge = 5.0

    return RateSchedule(
        rates=[base_rate, tou_rate],
        monthly_fixed_charge=fixed_charge,
    )


@pytest.fixture
def seasonal_rate_schedule():
    """Create a rate schedule with seasonal and time-of-use rates to test with."""
    non_seasonal_base_rate = Rate(volumetric_rate=0.38811, name="non seasonal base")
    non_seasonal_tou_rate = Rate(volumetric_rate=0.41646, tou_start_hour=16, tou_end_hour=21, name="non seasonal TOU")
    seasonal_base_rate = Rate(volumetric_rate=0.43573, seasonal_start_month=6, seasonal_end_month=10)
    seasonal_time_of_use_rate = Rate(
        volumetric_rate=0.51917, tou_start_hour=16, tou_end_hour=21, seasonal_start_month=6, seasonal_end_month=10
    )
    fixed_charge = 5.25

    return RateSchedule(
        rates=[non_seasonal_base_rate, non_seasonal_tou_rate, seasonal_base_rate, seasonal_time_of_use_rate],
        monthly_fixed_charge=fixed_charge,
    )


@pytest.fixture
def holiday_rate_schedule():
    """Rate schedule with a different flat rate on weekends and holidays."""
    non_seasonal_base_rate = Rate(volumetric_rate=0.38811, name="non seasonal base")
    non_seasonal_tou_rate = Rate(
        volumetric_rate=0.41646,
        tou_start_hour=16,
        tou_end_hour=21,
        name="non seasonal TOU",
        days_applicable=DaysApplicable.WEEKDAYS,
    )
    seasonal_base_rate = Rate(
        volumetric_rate=0.43573, seasonal_start_month=6, seasonal_end_month=10, days_applicable=DaysApplicable.WEEKDAYS
    )
    seasonal_time_of_use_rate = Rate(
        volumetric_rate=0.51917,
        tou_start_hour=16,
        tou_end_hour=21,
        seasonal_start_month=6,
        seasonal_end_month=10,
        days_applicable=DaysApplicable.WEEKDAYS,
    )
    fixed_charge = 5.25

    holiday_rate = Rate(volumetric_rate=0.2, name="holiday rate", days_applicable=DaysApplicable.WEEKENDS_HOLIDAYS)

    return RateSchedule(
        rates=[
            non_seasonal_base_rate,
            non_seasonal_tou_rate,
            seasonal_base_rate,
            seasonal_time_of_use_rate,
            holiday_rate,
        ],
        monthly_fixed_charge=fixed_charge,
    )
