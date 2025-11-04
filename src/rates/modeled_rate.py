"""ModeledRate class, which applies a rate schedule to a set of homes from ResStock."""

from enum import Enum
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USThanksgivingDay,
    sunday_to_monday,
)

from rates.rate_structure import EmissionsSchedule, RateSchedule, Schedule


class FixedChargeOption(Enum):
    """Enum for fixed charge modeling options."""

    NEVER = "never"
    ALWAYS = "always"
    IF_CONSUMPTION = "if_consumption"


class ModeledRate:
    """
    Model the energy bills for a given fuel for a set of building load profiles under a given rate structure.

    Attributes
    ----------
        monthly_fixed_charge : float
            Fixed charge applied monthly.
        building_ids : np.array
            list of ResStock building ids corresponding to the current analysis
        rate_matrix : np.array
            Rate by [month, hour]
        load_profile_matrix : np.array
            Matrix of hourly fuel consumption for each [building, month, hour, day].
        n_dim_by_axis : dict
            Dictionary containing the dimensions for each axis of load_profile_matrix.
        axis_order : list
            List of axis names of load_profile_matrix in order.
        _bills_by_building_month_hour : np.array
            Matrix of cost by [building, month, hour] summed over all days in the month.
        name : str
            Name of rate. Default is "Unnamed"
        fuel : str
            Name of the fuel

    """

    N_MONTHS_IN_YEAR = 12
    N_HOURS_IN_DAY = 24
    N_MAX_DAYS_IN_MONTH = 31

    @classmethod
    def create_modeled_rate_per_upgrade(cls, load_profile_upgrades_df, rate_schedule, **kwargs):
        """
        Create a new ModeledRate for each upgrade present in load_profile_df.

        Parameters
        ----------
        load_profile_upgrades_df: pd.DataFrame
            DataFrame containing the hourly electric load profile with columns 'timestamp', 'building_id',
            'upgrade_id', and '{fuel}_kwh'.
        rate_schedule: Schedule
            A rate schedule - may be any subclass of Schedule, including
            RateSchedule, FlatRateSchedule, or EmissionsSchedule.
        kwargs
            Other keyword args for the ModeledRate constructor, including `name` and `fuel`.

        Returns
        -------
        dict[str, ModeledRate]
            A dictionary that maps from each upgrade ID (as a string) in load_profile_upgrades_df to a new ModeledRate object
            for the load profile of that upgrade.
        """
        upgrade_ids = load_profile_upgrades_df["upgrade_id"].unique()
        upgrade_to_modeled_rate = {
            str(upgrade_id): cls(
                load_profile_upgrades_df[load_profile_upgrades_df.upgrade_id == upgrade_id],
                rate_schedule,
                **kwargs,
            )
            for upgrade_id in upgrade_ids
        }
        return upgrade_to_modeled_rate

    def __init__(
        self,
        load_profile_df: pd.DataFrame,
        rate_schedule: Schedule,
        name: str = "Unnamed",
        fuel: str = "electricity",
    ):
        """
        Create a ModeledRate from the components of a rate schedule.

        Parameters
        ----------
        load_profile_df
            DataFrame containing the hourly load profile with columns:
            'building_id', 'month', 'day', 'hour', and '{self.fuel}_kwh'.
        rate_schedule
            A rate schedule - may be any subclass of Schedule, including
            RateSchedule, FlatRateSchedule, or EmissionsSchedule.
            Note: Weekend/holiday rates will be ignored - use ModeledRateWithWeekendHoliday
                to incorporate them.
        name
            Name of rate. Default is "Unnamed"
        fuel
            Name of the fuel
        """
        self.name = name
        self.fuel = fuel
        # -- Rate attributes -- #
        # fixed charge ($/month)
        self.monthly_fixed_charge = rate_schedule.monthly_fixed_charge

        # rates in $/kwh by [month, hour]
        self.rate_matrix = rate_schedule.rate_matrix()

        # -- Load profile attributes -- #
        self.building_ids = load_profile_df.building_id.unique()
        self.n_dim_by_axis = {
            "building_index": len(self.building_ids),
            "month": self.N_MONTHS_IN_YEAR,
            "hour": self.N_HOURS_IN_DAY,
            "day": self.N_MAX_DAYS_IN_MONTH,
        }
        self.axis_order = list(self.n_dim_by_axis.keys())
        # usage matrix of kwh by [building, month, hour, day]
        self.load_profile_matrix = self.build_load_profile_matrix(load_profile_df)

        # -- Cached calculated bill attributes -- #
        # Bills aggregated by [building, month, hour] which is expensive to compute
        self._bills_by_building_month_hour = None

    def __str__(self):
        """To string method."""
        return "_".join(self.name.lower().split())

    def build_load_profile_matrix(self, load_profile_df: pd.DataFrame) -> np.array:
        """
        Create a 4D load profile matrix from the given load profile DataFrame.

        Parameters
        ----------
        load_profile_df
            DataFrame containing the hourly load profile with columns:
            'building_id', 'month', 'day', 'hour', and '{self.fuel}_kwh'.

        Returns
        -------
        np.ndarray
            A 4D numpy array with the fuel consumption (kwh) with shape [building, month, hour, day]
        """
        load_profile_df = load_profile_df.copy()
        # Add a 0-indexed unique consecutive index for each building
        load_profile_df["building_index"] = pd.factorize(load_profile_df["building_id"])[0]
        # Convert day and month to be 0-indexed
        load_profile_df["day"] -= 1
        load_profile_df["month"] -= 1
        # Initialize a 4D numpy array for [building, month, hour, day] filled with NaN
        load_profile_matrix = np.full(list(self.n_dim_by_axis.values()), fill_value=np.nan)
        # Assign the values directly to the matrix based on the [building, month, hour, day]
        # load_profile_matrix[
        #     *[load_profile[x].values for x in self.axis_order]
        # ] = load_profile["electricity_kwh"].values
        load_profile_matrix[
            load_profile_df["building_index"],
            load_profile_df["month"],
            load_profile_df["hour"],
            load_profile_df["day"],
        ] = load_profile_df[f"{self.fuel}_kwh"].values

        # Replace NaN with 0
        load_profile_matrix = np.nan_to_num(load_profile_matrix, 0)

        return load_profile_matrix

    @staticmethod
    def compute_bills_by_building_month_hour(rate_matrix: np.array, load_profile_matrix: np.array) -> np.array:
        """
        Apply volumetric rates to the load profile to compute cost by building, month, hour.

        Parameters
        ----------
        rate_matrix
            A matrix of rates ($/kwh) with shape [month, hour]
        load_profile_matrix
            A 4D numpy array with the fuel consumption (kwh) with shape [building, month, hour, day]

        Returns
        -------
        np.ndarray
            The calculated monthly bills ($) for each building with shape [building, month, hour]
        """
        # load profile (kwh) [building, month, hour, day] * rate_matrix ($/kwh) [month, hour]
        # summed over [day] = cost ($) for each [building, month]
        # Use einsum to multiply along the shared axes (month, hour) and sum over (hour) in one step
        return np.einsum("bmhd,mh->bmh", load_profile_matrix, rate_matrix)

    @property
    def bills_by_building_month_hour(self) -> np.array:
        """
        Calculate cost by (building, month, hour) without fixed charges.

        Applies the volumetric rate matrix to the hourly electric consumption matrix.

        Returns
        -------
        np.ndarray
            The total cost summed over days by [building, month, hour]
        """
        if self._bills_by_building_month_hour is None:
            self._bills_by_building_month_hour = self.compute_bills_by_building_month_hour(
                load_profile_matrix=self.load_profile_matrix, rate_matrix=self.rate_matrix
            )
        return self._bills_by_building_month_hour

    # TODO: add baseline allowance/credit as part of the rateschedule/ratemodel constructor
    def apply_baseline_credit(
        self,
        baseline_credit_dollars_per_kwh: float,
        baseline_allowance_kwh_per_day: float,
    ):
        """
        Apply the baseline allowance credit to the bills by [building, month, hour].

        This applies a credit of `baseline_credit_dollars_per_kwh` for each kWh used by a household,
        for up to `baseline_allowance_kwh_per_day` per day, adjusting the values in
        self._bills_by_building_month_hour.

        Parameters
        ----------
        baseline_credit_dollars_per_kwh
            The baseline credit in dollars per kWh.
        baseline_allowance_kwh_per_day
            The baseline allowance in kwh per day.
        """
        # kwh per [building, month, day]
        daily_load_by_building_month_day = self.load_profile_matrix.sum(axis=self.axis_order.index("hour"))
        # daily credit from baseline allowance in dollar per [building, month, day] -- this credit gets subtracted off the bill
        daily_baseline_credit_by_building_month_day = (
            daily_load_by_building_month_day.clip(0, baseline_allowance_kwh_per_day) * baseline_credit_dollars_per_kwh
        )
        # sum over days and diseggrate over hours to get average hourly credit per [building, month]
        average_hourly_baseline_credit_by_building_month = (
            daily_baseline_credit_by_building_month_day.sum(axis=-1) / self.N_HOURS_IN_DAY
        )
        # subtract credit from bills by [building, month, hour]
        self._bills_by_building_month_hour = self.bills_by_building_month_hour - np.expand_dims(
            average_hourly_baseline_credit_by_building_month, axis=-1
        )

    def calculate_monthly_bills(
        self,
        fixed_charge_option: FixedChargeOption = FixedChargeOption.IF_CONSUMPTION,
    ) -> np.array:
        """
        Calculate the monthly bills by building, month.

        Parameters
        ----------
        fixed_charge_option : FixedChargeOption
            How to include the fixed charge:
                * FixedChargeOption.NEVER: fixed charge not included in monthly bill for any buildings
                * FixedChargeOption.ALWAYS: fixed charge included in monthly bill for all buildings
                * FixedChargeOption.IF_CONSUMPTION (default): fixed charge included in monthly bill if annual consumption
                for that building is positive.

        Returns
        -------
        np.array
            The calculated monthly bills ($) for each building with shape [building, month]
        """
        # sum costs by [building, month, hour] across [month] to get cost for each [building, month]
        monthly_volumetric_cost = self.bills_by_building_month_hour.sum(axis=(self.axis_order.index("hour")))

        if fixed_charge_option == FixedChargeOption.NEVER:
            return monthly_volumetric_cost
        elif fixed_charge_option == FixedChargeOption.ALWAYS:
            return monthly_volumetric_cost + self.monthly_fixed_charge
        if fixed_charge_option == FixedChargeOption.IF_CONSUMPTION:
            annual_load_per_building = self.load_profile_matrix.sum(
                axis=(self.axis_order.index("hour"), self.axis_order.index("month"), self.axis_order.index("day"))
            )
            fixed_charge = np.where(annual_load_per_building > 0, self.monthly_fixed_charge, 0.0)  # [building]
            return monthly_volumetric_cost + fixed_charge[:, np.newaxis]  # [building, month]

    def calculate_annual_bills(
        self,
        fixed_charge_option: FixedChargeOption = FixedChargeOption.IF_CONSUMPTION,
    ) -> np.ndarray:
        """
        Calculate the annual bills by building.

        Parameters
        ----------
        fixed_charge_option : FixedChargeOption
            How to include the fixed charge:
                * FixedChargeOption.NEVER: fixed charge not included in monthly bill for any buildings
                * FixedChargeOption.ALWAYS: fixed charge included in monthly bill for all buildings
                * FixedChargeOption.IF_CONSUMPTION (default): fixed charge included in monthly bill if annual consumption
                for that building is positive.

        Returns
        -------
        np.ndarray
            The vector of calculated annual bills for each building of length [building]
        """
        # sum monthly costs across [month] to get cost for each [building]
        monthly_volumetric_cost = self.calculate_monthly_bills(fixed_charge_option=fixed_charge_option)
        annual_volumetric_cost = monthly_volumetric_cost.sum(self.axis_order.index("month"))

        return annual_volumetric_cost

    def to_long_format(self, upgrade_id: int, granularity: str = "annual") -> pd.DataFrame:
        """
        Convert rate calculation results to standardized long format.

        Returns
        -------
        pd.DataFrame
            DataFrame in long format with columns:
            - Always: (building_id, energy, cost, upgrade_id, fuel, rate_name)
            - For monthly: adds 'month'
            - For hourly: adds 'hour'
            - For monthly_hourly: adds both 'month' and 'hour'
        """
        # Common metadata for all rows
        n_buildings = len(self.building_ids)
        common_metadata = {"upgrade_id": upgrade_id, "fuel": self.fuel, "rate_name": self.name}

        if granularity == "annual":
            return pd.DataFrame(
                {
                    "building_id": self.building_ids,
                    "energy": self.load_profile_matrix.sum(axis=(1, 2, 3)),
                    "cost": self.calculate_annual_bills(),
                    **common_metadata,
                }
            )

        elif granularity == "monthly":
            return pd.DataFrame(
                {
                    "building_id": np.repeat(self.building_ids, 12),
                    "month": np.tile(np.arange(1, 13), n_buildings),
                    "energy": self.load_profile_matrix.sum(axis=(2, 3)).flatten(),
                    "cost": self.calculate_monthly_bills().flatten(),
                    **common_metadata,
                }
            )

        elif granularity == "hourly":
            return pd.DataFrame(
                {
                    "building_id": np.repeat(self.building_ids, 12 * 24),
                    "hour": np.tile(np.arange(24), 12 * n_buildings),
                    "energy": self.load_profile_matrix.sum(axis=3).flatten(),
                    "cost": self.bills_by_building_month_hour.flatten(),
                    **common_metadata,
                }
            )

        elif granularity == "monthly_hourly":
            df = pd.DataFrame(
                {
                    "building_id": np.repeat(self.building_ids, 12 * 24),
                    "month": np.tile(np.repeat(np.arange(1, 13), 24), n_buildings),
                    "hour": np.tile(np.arange(24), 12 * n_buildings),
                    "energy": self.load_profile_matrix.sum(axis=3).flatten(),
                    "cost": self.bills_by_building_month_hour.flatten(),
                    **common_metadata,
                }
            )
            # create an additional column with the mean energy and cost per day rather than the sum over all days
            # First create a dictionary of days in each month
            days_in_month = {
                1: 31,  # January
                2: 28,  # February (assuming not leap year)
                3: 31,  # March
                4: 30,  # April
                5: 31,  # May
                6: 30,  # June
                7: 31,  # July
                8: 31,  # August
                9: 30,  # September
                10: 31,  # October
                11: 30,  # November
                12: 31,  # December
            }

            # Add a column for days in month
            df["days_in_month"] = df["month"].map(days_in_month)
            # compute mean energy and cost per day for the given building, month and hour
            df["energy_daily_mean"] = df["energy"] / df["days_in_month"]
            df["cost_daily_mean"] = df["cost"] / df["days_in_month"]

            return df

        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    @staticmethod
    def plot_matrix_by_month_hour(matrix: np.ndarray, n_decimal_places: int = 2, plot_title: str = ""):
        """Plot a heatmap of costs per month and hour."""
        # Define hour and month labels
        hours = [f"{hour}:00" for hour in range(24)]
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            matrix,
            xticklabels=hours,
            yticklabels=months,
            annot=True,  # Annotate each cell with its value
            annot_kws={"size": 8},
            fmt=f".{n_decimal_places}f",  # Format annotations with 0 decimal places
            cmap="viridis",  # Use your preferred colormap
            cbar_kws={"label": "Value"},  # Add a colorbar label
            linewidths=0.2,  # Add gridlines
            linecolor="#D3D3D3",  # Color of the gridlines
            cbar=False,
            ax=ax,
        )

        # Label axes
        ax.set_xlabel("Hours")
        ax.set_ylabel("Months")
        ax.set_title(plot_title)
        return fig


class ModeledRateWithWeekendHoliday(ModeledRate):
    """
    Model the electriciy bills for a set of building load profiles under a rate structure with differentiated rates for weekends/holidays vs weekdays.

    Attributes
    ----------
        rate_matrix_weekend_holiday : np.array
            Rates for holidays/weekends by [month, hour].
        holiday_calendar
            Calendar for utility holidays.
        weekend_holiday_mask : np.array
            Boolean mask by indicating whether the day in the given [month, hour] is a weekend or holiday.
        load_profile_matrix_weekday_non_holiday : np.array
            Matrix of hourly fuel consumption for each [building, month, hour, day] where values for weekends and holidays are 0.
        load_profile_matrix_weekend_holiday : np.array
            Matrix of hourly fuel consumption for each [building, month, hour, day] where values for weekday non-holidays are 0.
        _weekday_non_holiday_bills_by_building_month_hour : np.array
            Matrix of cost by [building, month, hour] summed over all weekday non-holiday days in the month.
        _weekend_holiday_bills_by_building_month_hour : np.array
            Matrix of cost by [building, month, hour] summed over all weekends and holidays days in the month.
        data_year
            The year of the data used for calculating holidays. Defaults to 2007, which is the year of ResStock TMY data.

    See ModeledRate for superclass attributes.

    """

    # Define NERC (North American Electric Reliability Corporation) Holidays
    # Source : https://www.lawinsider.com/contracts/Mm6qYthNf#nerc-holiday
    class NERCHolidays(AbstractHolidayCalendar):
        """NERC (North American Electric Reliability Corporation) Holiday Calendar."""

        rules = [
            Holiday("New Years Day", month=1, day=1, observance=sunday_to_monday),
            USMemorialDay,
            Holiday("Independence Day", month=7, day=4, observance=sunday_to_monday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas Day", month=12, day=25, observance=sunday_to_monday),
        ]

    def __init__(
        self,
        load_profile_df: pd.DataFrame,
        rate_schedule: Schedule,
        holiday_calendar: AbstractHolidayCalendar = NERCHolidays(),
        data_year=2007,
        name: str = "Unnamed",
        fuel: str = "electricity",
    ):
        """
        Create a ModeledRateWithWeekendHoliday from the components of a rate schedule.

        Parameters
        ----------
        load_profile_df
            DataFrame containing the hourly load profile with columns:
            'building_id', 'month', 'day', 'hour', and '{self.fuel}_kwh'.
        rate_schedule
            A rate schedule - may be any subclass of Schedule
            Note: Not all schedule types support different weekday vs weekend/holiday rates.
        holiday_calendar
            Calendar for utility holidays. Defaults to NERCHolidays().
        data_year
            The year of the data used for calculating holidays. Defaults to 2007, which is the year of ResStock TMY data.
        name
            Name of rate. Default is "Unnamed"
        fuel
            Name of the fuel

        """
        super().__init__(load_profile_df, rate_schedule, name=name, fuel=fuel)

        self.holiday_calendar = holiday_calendar
        self.data_year = data_year

        # -- Rate attributes -- #
        # holiday/weekend rates in $/kwh by [month, hour]
        self.rate_matrix_weekend_holiday = rate_schedule.rate_matrix(weekend_holiday=True)

        # -- Load profile attributes -- #
        self.weekend_holiday_mask = self.build_weekend_holiday_mask(load_profile_df)
        # expand the mask to the load profile matrix dimensions
        weekend_holiday_mask_expanded_dims = self.expand_matrix_to_load_profile_dimensions(
            matrix=self.weekend_holiday_mask, matrix_axis_names=["month", "day"]
        )
        # matrix of kwh by [building, month, hour, day] where values for weekday non-holidays are set to 0
        self.load_profile_matrix_weekend_holiday = np.where(
            weekend_holiday_mask_expanded_dims, self.load_profile_matrix, 0
        )
        # matrix of kwh by [building, month, hour, day] where values for weekends and holidays are set to 0
        self.load_profile_matrix_weekday_non_holiday = np.where(
            ~weekend_holiday_mask_expanded_dims, self.load_profile_matrix, 0
        )

        # -- Cached calculated bill attributes -- #
        # Bills aggregated by [building, month, hour]
        self._weekday_non_holiday_bills_by_building_month_hour = None
        self._weekend_holiday_bills_by_building_month_hour = None

    def build_weekend_holiday_mask(self, load_profile_df) -> np.array:
        """
        Build bool mask of shape [month, day] indicating whether each day of the year is a weekend or holiday.

        Parameters
        ----------
        load_profile_df : pd.DataFrame
            DataFrame containing the hourly load profile with columns:
            'building_id', 'month', 'day', 'hour', and '{self.fuel}_kwh'.

        Returns
        -------
        np.array
            A boolean mask of shape [month, day] indicating whether each day of the year is a weekend or holiday.
        """
        # get the first and last date in the load profile
        # we can just select one sample building to do this so that the pd.to_datetime is as cheap as possible
        prototype_df = load_profile_df[(load_profile_df["building_id"] == load_profile_df.iloc[0].building_id)].copy()
        # assign a dummy year that is the same as the TMY data year
        timestamps = pd.to_datetime(prototype_df.assign(year=self.data_year)[["year", "month", "day", "hour"]])

        # construct a dataframe of one day per date in the load profile date range with weekend/holiday status per date
        date_range = pd.date_range(start=timestamps.min().date(), end=timestamps.max().date())
        holiday_weekend_df = pd.DataFrame()
        holiday_weekend_df["date"] = date_range
        holiday_weekend_df["month"] = holiday_weekend_df.date.dt.month - 1
        holiday_weekend_df["day"] = holiday_weekend_df.date.dt.day - 1

        # get holiday calendar spanning from first to last date
        holidays = self.holiday_calendar.holidays(start=date_range.min(), end=date_range.max())
        # calculate if each day of the year is either a weekend or a holiday
        holiday_weekend_df["is_weekend"] = holiday_weekend_df.date.dt.dayofweek >= 5
        holiday_weekend_df["is_holiday"] = holiday_weekend_df.date.isin(holidays)
        holiday_weekend_df["is_weekend_or_holiday"] = holiday_weekend_df.is_holiday | holiday_weekend_df.is_weekend

        # create mask of [month, day] where weekend_holiday_matrix[i,j] = True indicates that the jth day of the ith month is a weekend or holiday
        weekend_holiday_msk_axes = ["month", "day"]  # define which axes the mask applies to
        weekend_holiday_msk = np.full(
            tuple(self.n_dim_by_axis[axis_name] for axis_name in weekend_holiday_msk_axes), dtype=bool, fill_value=False
        )  # initialize matrix
        weekend_holiday_msk[  # fill matrix with bool values from dataframe
            holiday_weekend_df["month"],
            holiday_weekend_df["day"],
        ] = holiday_weekend_df.is_weekend_or_holiday.values

        return weekend_holiday_msk

    def expand_matrix_to_load_profile_dimensions(self, matrix: np.array, matrix_axis_names: List[str]) -> np.array:
        """
        Expand a matrix with a given set of named axes (e.g, ['month', 'day']) to be the dimensions of the load profile matrix.

        Parameters
        ----------
        matrix : np.array
            matrix with a subset of the dimensions of the load profile matrix
        matrix_axis_names : list[str]
            the ordered names of the axes of the passed matrix, which shuld be a subset of model.axis_order

        Returns
        -------
        np.array
            The calculated monthly bills ($) for each building with shape [building, month]
        """
        # insert `np.newaxis` (None) for all dimensons not in the matrix
        slicer = tuple(np.newaxis if axis not in matrix_axis_names else slice(None) for axis in self.axis_order)
        return matrix[slicer]

    @property
    def weekday_non_holiday_bills_by_building_month_hour(self) -> np.array:
        """
        Calculate cost by (building, month, hour) for weekdays and non-holidays.

        Returns
        -------
        np.array
            The total cost summed over weekday non-holiday days in the month by [building, month, hour]
        """
        if self._weekday_non_holiday_bills_by_building_month_hour is None:
            self._weekday_non_holiday_bills_by_building_month_hour = self.compute_bills_by_building_month_hour(
                load_profile_matrix=self.load_profile_matrix_weekday_non_holiday, rate_matrix=self.rate_matrix
            )
        return self._weekday_non_holiday_bills_by_building_month_hour

    @property
    def weekend_holiday_bills_by_building_month_hour(self) -> np.array:
        """
        Calculate cost by (building, month, hour) for weekends and holidays.

        Returns
        -------
        np.array
            The total cost summed over weekends and holiday days in the month by [building, month, hour]
        """
        if self._weekend_holiday_bills_by_building_month_hour is None:
            self._weekend_holiday_bills_by_building_month_hour = self.compute_bills_by_building_month_hour(
                load_profile_matrix=self.load_profile_matrix_weekend_holiday,
                rate_matrix=self.rate_matrix_weekend_holiday,
            )
        return self._weekend_holiday_bills_by_building_month_hour

    @property
    def bills_by_building_month_hour(self) -> np.array:
        """
        Calculate cost by (building, month, hour).

        Returns
        -------
        np.array
            The total cost summed over all days in the month by [building, month, hour]
        """
        if self._bills_by_building_month_hour is None:
            self._bills_by_building_month_hour = (
                self.weekday_non_holiday_bills_by_building_month_hour
                + self.weekend_holiday_bills_by_building_month_hour
            )
        return self._bills_by_building_month_hour


# TODO: maybe fold these into the class?
def compute_total_energy_costs(df: pd.DataFrame):
    """
    Compute total energy costs across all fuels for each (building, upgrade, electric rate).

    Note that this assumes there is only one 'rate_name' per 'fuel' for all fossil fuels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing energy cost data with columns:
        ['building_id', 'energy', 'cost', 'upgrade_id', 'fuel', 'rate_name', ...time_columns]

    Returns
    -------
    pandas.DataFrame
        The combined dataframe with original data plus total cost calculations

    """
    # Get non-electric fuels
    non_electric_fuels = [f for f in df["fuel"].unique() if f != "electricity"]

    # Get any time columns if df is at is sub-annual level
    time_columns = list(
        set(df.columns).difference(["building_id", "energy", "cost", "upgrade_id", "fuel", "rate_name"])
    )

    # Sum all NON-electric cost
    fossil_fuel_cost = (
        df[df["fuel"].isin(non_electric_fuels)]
        .groupby(["building_id", "upgrade_id", *time_columns], as_index=False)[["cost", "energy"]]
        .sum()
    )

    # Keep electricity cost with their rate_name
    electricity_cost = df[df["fuel"] == "electricity"][
        ["building_id", "upgrade_id", "rate_name", *time_columns, "cost", "energy"]
    ]

    # Merge electricity + fossil fuel cost
    # and fill null with 0s for upgrades with no non-electric cost
    total_df = electricity_cost.merge(
        fossil_fuel_cost,
        on=["building_id", "upgrade_id", *time_columns],
        how="left",
        suffixes=["_electricity", "_fossil_fuel"],
    ).fillna(0)

    # Calculate total cost
    total_df["cost"] = total_df["cost_electricity"] + total_df["cost_fossil_fuel"]
    total_df["energy"] = total_df["energy_electricity"] + total_df["energy_fossil_fuel"]
    total_df["fuel"] = "total"

    return total_df[df.drop(["fuel"], axis=1).columns]


def compute_savings(df_total: pd.DataFrame, baseline_upgrade_id: float = 0.01):
    """
    Compute savings over all fuels for each (building, upgrade, rate_name) as well as any time columns.

    Savings are calculated by comparing each non-baseline upgrade to the specified baseline.

    Parameters
    ----------
    df_total : pandas.DataFrame
        Input dataframe containing total energy cost data with columns:
        ['building_id', 'energy', 'cost', 'upgrade_id', 'rate_name', ...time_columns]
    baseline_upgrade_id : float
        Upgrade ID for baseline. Defaults to 0.01 which is baseline for ResStock 2024.2.

    Returns
    -------
    pandas.DataFrame
        Dataframe with total savings over all fuels.

    """
    # Get any time columns if df is at is sub-annual level
    time_columns = list(
        set(df_total.columns).difference(["building_id", "energy", "cost", "upgrade_id", "fuel", "rate_name"])
    )
    # Get every combination of (upgrade & upgrade rate x baseline rate)
    df_baseline = df_total[df_total.upgrade_id == baseline_upgrade_id].drop("upgrade_id", axis=1)
    df_upgrade = df_total[df_total.upgrade_id != baseline_upgrade_id]
    df_combined = df_upgrade.merge(df_baseline, on=["building_id", *time_columns], suffixes=["", "_baseline"])

    # # Subtract upgrade from baseline to compute savings
    df_combined["energy_savings"] = df_combined["energy_baseline"] - df_combined["energy"]
    df_combined["cost_savings"] = df_combined["cost_baseline"] - df_combined["cost"]
    df_combined["has_positive_savings"] = df_combined.cost_savings > 0
    return df_combined
