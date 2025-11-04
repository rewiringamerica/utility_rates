"""Tests for Rate and RateSchedule classes."""

from rates.rate_structure import DaysApplicable, Rate


class TestRate:
    """Test suite for the Rate class."""

    def test_basic_rate_initialization(self):
        """Test basic initialization with required parameters."""
        rate = Rate(0.12, name="Basic Rate")
        assert rate.volumetric_rate == 0.12
        assert rate.name == "Basic Rate"
        assert rate.tou_start_hour == 0
        assert rate.tou_end_hour == 24
        assert rate.seasonal_start_month == 1
        assert rate.seasonal_end_month == 13
        assert rate.days_applicable == DaysApplicable.ALL

    def test_rate_days_applicable(self):
        """Test initialization of a rate that only applies to weekdays."""
        rate = Rate(0.12, days_applicable=DaysApplicable.WEEKDAYS)
        assert rate.days_applicable == DaysApplicable.WEEKDAYS

    def test_default_mask_array(self):
        """Test that mask array is all True for default initialization."""
        rate = Rate(0.12)
        mask = rate.mask_array
        assert mask.shape == (12, 24)
        assert mask.all()  # All values should be True

    def test_tou_hour_range(self):
        """Test that TOU hours are correctly applied in the mask."""
        rate = Rate(0.12, tou_start_hour=8, tou_end_hour=20)
        mask = rate.mask_array
        # All months should have the same pattern
        # Hours 0-7 should be False
        assert not mask[:, :8].any()
        # Hours 8-19 should be True
        assert mask[:, 8:20].all()
        # Hours 20-23 should be False
        assert not mask[:, 20:].any()

    def test_seasonal_range(self):
        """Test that seasonal months are correctly applied in the mask."""
        rate = Rate(0.12, seasonal_start_month=5, seasonal_end_month=9)
        mask = rate.mask_array
        # Months 1-4 (0-3 in zero-index) should be all False
        assert not mask[:4, :].any()
        # Months 5-8 (4-7 in zero-index) should be all True
        assert mask[4:8, :].all()
        # Months 9-12 (8-11 in zero-index) should be all False
        assert not mask[8:, :].any()

    def test_combined_tou_and_seasonal(self):
        """Test combined TOU and seasonal parameters."""
        rate = Rate(0.12, tou_start_hour=10, tou_end_hour=18, seasonal_start_month=6, seasonal_end_month=9)
        mask = rate.mask_array

        # Check months outside season (0-5, 8-11)
        assert not mask[0:5, :].any()
        assert not mask[8:11, :].any()

        # Check months in season (6-8)
        # Hours 0-9 should be False
        assert not mask[5:8, 0:10].any()
        # Hours 10-17 should be True
        assert mask[5:8, 10:18].all()
        # Hours 18-23 should be False
        assert not mask[:, 18:].any()

    def test_full_year_rate(self):
        """Test rate defined for full year."""
        # Full year, full day
        rate = Rate(0.12, seasonal_start_month=1, seasonal_end_month=13, tou_start_hour=0, tou_end_hour=24)
        assert rate.mask_array.all()

    def test_single_month_hour_rate(self):
        """Test rate defined for a single hour/month."""
        # Single month, single hour
        rate = Rate(0.12, seasonal_start_month=6, seasonal_end_month=7, tou_start_hour=12, tou_end_hour=13)
        mask = rate.mask_array
        assert mask[5, 12]  # Only June (5), hour 12 should be True
        assert not mask[5, 11] and not mask[5, 13]  # Adjacent hours should be False
        assert not mask[4, :].any() and not mask[6, :].any()  # Other months should be False

    def test_rate_bridging_new_years(self):
        """Test rate that crosses new years."""
        rate = Rate(0.12, seasonal_start_month=11, seasonal_end_month=2)
        mask = rate.mask_array

        assert mask[0:1, :].all()
        assert mask[11:12, :].all()
        assert not mask[2:10, :].any()

    def test_rate_bridging_midnight(self):
        """Test rate that crosses midnight."""
        rate = Rate(0.12, tou_start_hour=23, tou_end_hour=1)
        mask = rate.mask_array

        assert mask[:, :1].all()
        assert mask[:, 23:].all()
        assert not mask[:, 2:22].any()
