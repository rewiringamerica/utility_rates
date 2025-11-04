"""Tests for the various Schedule classes."""

import numpy as np
import pytest

from rates.rate_structure import (
    DaysApplicable,
    EmissionsSchedule,
    FlatRateSchedule,
    Rate,
    RateIndexSchedule,
    RateSchedule,
)


def test_flat_rate_schedule():
    """Test flat rate schedule."""
    flat_rate = FlatRateSchedule(0.10, 20)
    assert flat_rate.monthly_fixed_charge == 20
    assert flat_rate.rate_matrix().shape == (12, 24)
    assert (flat_rate.rate_matrix() == 0.10).all()
    # Weekend/holiday rates are the same
    assert (flat_rate.rate_matrix(weekend_holiday=True) == flat_rate.rate_matrix(weekend_holiday=False)).all()


def test_emissions_schedule():
    """Test emissions schedule."""
    rate_matrix = np.ones((12, 24), dtype=int)
    rate_matrix[:, 12:18] = 2  # Peak hours is 2, off peak is 1

    emissions = EmissionsSchedule(rate_matrix)
    assert emissions.monthly_fixed_charge == 0
    assert (emissions.rate_matrix() == rate_matrix).all()
    # Weekend/holiday rates are the same
    assert (emissions.rate_matrix(weekend_holiday=True) == rate_matrix).all()


def test_rate_index_schedule():
    """Test a rate index schedule."""
    # Create a TOU schedule where hours 12-17 (noon-5pm) are peak
    rate_schedule_matrix = np.zeros((12, 24), dtype=int)
    rate_schedule_matrix[:, 12:18] = 1  # Peak hours is 1, off peak is 0

    schedule = RateIndexSchedule(
        rate_schedule_matrix=rate_schedule_matrix,
        volumetric_rates=np.array([0.10, 0.20]),  # Off-peak and peak rates
        monthly_fixed_charge=5.0,
    )

    rate_matrix = schedule.rate_matrix()
    assert schedule.monthly_fixed_charge == 5.0
    assert np.all(rate_matrix[:, 12:18] == 0.20)
    assert np.all(rate_matrix[:, :12] == 0.10)
    assert np.all(rate_matrix[:, 18:] == 0.10)

    # Test referencing rates that don't exist in volumetric_rates
    with pytest.raises(ValueError):
        schedule = RateIndexSchedule(
            rate_schedule_matrix=rate_schedule_matrix,
            volumetric_rates=np.array([0.10]),
            monthly_fixed_charge=5.0,
        )


def compare_rate_schedules(rate1: RateSchedule, rate2: RateSchedule):
    """Assert that two RateSchedules are the same."""
    # Check the attributes of each object.
    assert rate1.monthly_fixed_charge == rate2.monthly_fixed_charge
    assert rate1.rates == rate2.rates

    # Also check the schedule arrays they produce.
    # Weekday rates
    old_schedule, old_rates = rate1.rate_schedule
    new_schedule, new_rates = rate2.rate_schedule
    assert new_rates == old_rates
    assert (new_schedule == old_schedule).all()

    # Weekend/holiday rates
    old_schedule, old_rates = rate1.weekend_holiday_rate_schedule
    new_schedule, new_rates = rate2.weekend_holiday_rate_schedule
    assert new_rates == old_rates
    assert (new_schedule is None and old_schedule is None) or (new_schedule == old_schedule).all()


def test_to_json(seasonal_rate_schedule):
    """Test converting rate schedules to and from JSON."""
    json_str = seasonal_rate_schedule.to_json()
    assert (
        json_str
        == """{"rates": [{"volumetric_rate": 0.38811, "name": "non seasonal base", "tou_start_hour": 0, "tou_end_hour": 24, "seasonal_start_month": 1, "seasonal_end_month": 13, "days_applicable": "all days"}, {"volumetric_rate": 0.41646, "name": "non seasonal TOU", "tou_start_hour": 16, "tou_end_hour": 21, "seasonal_start_month": 1, "seasonal_end_month": 13, "days_applicable": "all days"}, {"volumetric_rate": 0.43573, "name": null, "tou_start_hour": 0, "tou_end_hour": 24, "seasonal_start_month": 6, "seasonal_end_month": 10, "days_applicable": "all days"}, {"volumetric_rate": 0.51917, "name": null, "tou_start_hour": 16, "tou_end_hour": 21, "seasonal_start_month": 6, "seasonal_end_month": 10, "days_applicable": "all days"}], "monthly_fixed_charge": 5.25}"""
    )

    new_schedule = RateSchedule.from_json(json_str)

    compare_rate_schedules(seasonal_rate_schedule, new_schedule)


def test_holiday_rate_to_json(holiday_rate_schedule):
    """Test converting a rate schedule with weekend/holiday rates to and from JSON."""
    json_str = holiday_rate_schedule.to_json()
    assert (
        json_str
        == """{"rates": [{"volumetric_rate": 0.38811, "name": "non seasonal base", "tou_start_hour": 0, "tou_end_hour": 24, "seasonal_start_month": 1, "seasonal_end_month": 13, "days_applicable": "all days"}, {"volumetric_rate": 0.41646, "name": "non seasonal TOU", "tou_start_hour": 16, "tou_end_hour": 21, "seasonal_start_month": 1, "seasonal_end_month": 13, "days_applicable": "weekdays only"}, {"volumetric_rate": 0.43573, "name": null, "tou_start_hour": 0, "tou_end_hour": 24, "seasonal_start_month": 6, "seasonal_end_month": 10, "days_applicable": "weekdays only"}, {"volumetric_rate": 0.51917, "name": null, "tou_start_hour": 16, "tou_end_hour": 21, "seasonal_start_month": 6, "seasonal_end_month": 10, "days_applicable": "weekdays only"}, {"volumetric_rate": 0.2, "name": "holiday rate", "tou_start_hour": 0, "tou_end_hour": 24, "seasonal_start_month": 1, "seasonal_end_month": 13, "days_applicable": "weekends/holidays only"}], "monthly_fixed_charge": 5.25}"""
    )

    new_schedule = RateSchedule.from_json(json_str)

    compare_rate_schedules(holiday_rate_schedule, new_schedule)


def test_to_json_file(holiday_rate_schedule, tmp_path):
    """Test converting rate schedules to and from JSON files."""
    json_filename = tmp_path / "schedule.json"
    holiday_rate_schedule.to_json_file(json_filename)

    new_rate_schedule = RateSchedule.from_json_file(json_filename)

    compare_rate_schedules(holiday_rate_schedule, new_rate_schedule)


def test_has_weekend_holiday_rates(seasonal_rate_schedule, holiday_rate_schedule):
    """Test the has_weekend_holiday_rates() method."""
    assert not seasonal_rate_schedule.has_weekend_holiday_rates()
    assert holiday_rate_schedule.has_weekend_holiday_rates()


def test_rate_matrix(simple_rate_schedule):
    """Test getting a rate matrix with rates ($/kWh) per [month, hour]."""
    rate_matrix = simple_rate_schedule.rate_matrix()
    assert (rate_matrix[5:8, 12:20] == 0.4).all()
    assert (rate_matrix[np.r_[0:5, 9:12], :] == 0.2).all()
    assert (rate_matrix[:, np.r_[0:12, 21:24]] == 0.2).all()


def test_rate_matrix_weekend_holidays(holiday_rate_schedule):
    """Test getting a weekend/holiday rate matrix with rates ($/kWh) per [month, hour]."""
    rate_matrix = holiday_rate_schedule.rate_matrix(weekend_holiday=False)
    assert (rate_matrix != 0.2).all()

    rate_matrix = holiday_rate_schedule.rate_matrix(weekend_holiday=True)
    assert (rate_matrix == 0.2).all()
