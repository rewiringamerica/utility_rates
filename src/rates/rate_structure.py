"""Defines classes for rates and rate schedules."""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]  # fmt: skip
HOURS = list(range(25))


class DaysApplicable(str, Enum):
    """Set of days a rate applies."""

    ALL = "all days"
    WEEKDAYS = "weekdays only"
    WEEKENDS_HOLIDAYS = "weekends/holidays only"


# Rates with any of these options apply on weekends and holidays
WEEKENDS_HOLIDAYS_OPTIONS = (DaysApplicable.ALL, DaysApplicable.WEEKENDS_HOLIDAYS)

# Rates with any of these options apply on weekdays
WEEKDAYS_OPTIONS = (DaysApplicable.ALL, DaysApplicable.WEEKDAYS)


@dataclass(eq=True)
class Rate:
    """
    A class representing a single annual rate with the potential to add time-of-use rates.

    Attributes
    ----------
        volumetric_rate : float
            the volumetric rate that the customer pays for electricity for this particular rate
        name : str
            the name of this particular rate
        tou_start_hour : int, optional
            the hour at which the customer starts paying a TOU volumetric rate (inclusive), in military 0 to 24.
        tou_end_hour : int, optional
            the hour at which the customer stops paying a TOU volumetric rate (exclusive), in military 0 to 24.
        seasonal_start_month : int, optional
            the month at which the customer starts paying a seasonal rate (inclusive).
        seasonal_end_month : int, optional
            the month at which the customer stops paying a seasonal rate (exclusive).
        days_applicable: DaysApplicable
            the days of the week the rate applies.
        mask_array : np.array
            a 12x24 boolean array that indicates where the volumetric rate contained in this instance
            of the Rate class should be applied
    """

    volumetric_rate: float
    name: Optional[str] = None
    tou_start_hour: Optional[int] = 0
    tou_end_hour: Optional[int] = 24
    seasonal_start_month: Optional[int] = 1
    seasonal_end_month: Optional[int] = 13
    days_applicable: DaysApplicable = DaysApplicable.ALL

    @property
    def mask_array(self) -> np.array:
        """
        Construct the mask array that will indicate where this rate's volumetric rate should be applied.

        Returns
        -------
        np.array
            a 12x24 boolean array that indicates where the volumetric rate contained in this
            instance of the Rate class should be applied
        """
        # Create a 12x24 array where all values are set to True
        # In the case where only given a base rate without time of use or seasonal changes, this will be the
        # value returned
        rate_array = np.zeros((12, 24), dtype=bool)

        # set month range to bridge over new years if end month is before start month
        # subtracting one from the input months to comply with python zero indexing.
        if self.seasonal_start_month > self.seasonal_end_month:
            seasonal_idx_range = np.r_[0 : self.seasonal_end_month - 1, self.seasonal_start_month - 1 : 12]
        else:
            seasonal_idx_range = np.r_[self.seasonal_start_month - 1 : self.seasonal_end_month - 1]

        # set hour range to bridge over midnight if end hour is before start hour
        if self.tou_start_hour > self.tou_end_hour:
            hourly_idx_range = np.r_[0 : self.tou_end_hour, self.tou_start_hour : 24]
        else:
            hourly_idx_range = np.r_[self.tou_start_hour : self.tou_end_hour]

        rate_array[np.ix_(seasonal_idx_range, hourly_idx_range)] = 1  # noqa: E203

        return rate_array


class Schedule(ABC):
    """
    Abstract class for different types of rate schedules.

    Specifies the methods needed to pass a schedule to a ModeledRate.
    """

    @abstractmethod
    def __init__(self, monthly_fixed_charge):
        self._monthly_fixed_charge = monthly_fixed_charge

    @abstractmethod
    def rate_matrix(self, weekend_holiday=False):
        """
        Get a rate matrix with rates (per kWh) for each [month, hour].

        This is typically ($/kWh), but may also represent other per-kWh measures, like CO2 emissions.

        Parameters
        ----------
        weekend_holiday:
            If true, get the rate matrix that applies on weekends and holiday.
            This parameter may be ignored if the schedule type doesn't distinguish between
                weekdays and weekends.
        """
        pass

    @property
    def monthly_fixed_charge(self):
        """The fixed charge applied monthly."""
        return self._monthly_fixed_charge


class FlatRateSchedule(Schedule):
    """
    A class that represents a simple flat volumetric rate structure.

    Attributes
    ----------
    volumetric_rate: The single volumetric rate that applies at all times, in USD per unit
        of fuel (e.g. $/kWh)
    _monthly_fixed_charge: float
        Fixed charge applied monthly, in USD.
    """

    N_MONTHS_IN_YEAR = 12
    N_HOURS_IN_DAY = 24

    def __init__(self, volumetric_rate, monthly_fixed_charge=0):
        super().__init__(monthly_fixed_charge)
        self.volumetric_rate = volumetric_rate

    def rate_matrix(self, weekend_holiday=False):
        """Get a rate matrix with rates ($/kWh) per [month, hour]."""
        return np.full((self.N_MONTHS_IN_YEAR, self.N_HOURS_IN_DAY), self.volumetric_rate)


class EmissionsSchedule(Schedule):
    """A class that represents time-varying emissions factors for a single fuel type."""

    def __init__(self, emissions_matrix):
        """
        Initialize an EmissionsSchedule.

        Parameters
        ----------
        emissions_matrix
           Numpy array of emissions factors (kg CO2e/kwh) by [month, hour]
        """
        super().__init__(monthly_fixed_charge=0)
        self._rate_matrix = emissions_matrix

    def rate_matrix(self, weekend_holiday=False):
        """Get a rate matrix with emission rates (kg CO2e/kWh) per [month, hour]."""
        return self._rate_matrix


class RateIndexSchedule(Schedule):
    """
    A schedule constructed directly from a rate schedule matrix and a list of volumetric rates.

    This is used when importing rate schedules from URDB. (https://openei.org/wiki/Utility_Rate_Database)
    """

    def __init__(self, volumetric_rates, rate_schedule_matrix, monthly_fixed_charge=0):
        """
        Create a RateIndexSchedule.

        Parameters
        ----------
        volumetric_rates
            Array of volumetric rates for fuel consumption.
        rate_schedule_matrix
            Matrix defining the rate schedule with rate index corresponding to `volumetric_rates` for each [month, hour].
        monthly_fixed_charge
            Fixed charge applied monthly. Default is 0.
        """
        super().__init__(monthly_fixed_charge)
        self.volumetric_rates = volumetric_rates
        self.rate_schedule_matrix = rate_schedule_matrix

        if (max_index := rate_schedule_matrix.max()) > (num_rates := len(volumetric_rates)) - 1:
            raise ValueError(
                f"Rate index {max_index} appears in rate_schedule_matrix, "
                f"but rate {num_rates - 1} is the last volumetric rate provided."
            )

    def rate_matrix(self, weekend_holiday=False):
        """Get a rate matrix with rates ($/kWh) per [month, hour]."""
        return self.volumetric_rates[self.rate_schedule_matrix]


class RateSchedule(Schedule):
    """
    A class that combines different rates to create a rate structure.

    Attributes
    ----------
        rates : List[Rate]
            a list of Rate objects. They need to be entered in layered order so that each entry
            into the list supercedes the rate that came before. This will often follow a logic of
            Non-Peak Season Base Rate, Non-Peak Season TOU Rate, Peak Season Base Rate, Peak Season TOU Rate.
        rate_schedule : np.array
            a 12x24 numpy array where each entry represents the volumetric electricity rate for
            a given hour on weekdays in a given month.
        weekend_holiday_rate_schedule : np.array
            a 12x24 numpy array where each entry represents the volumetric electricity rate for
            a given hour on weekends and holidays in a given month.
    """

    COLORS = [
        "lightblue",
        "lightpink",
        "gold",
        "lightgreen",
        "plum",
        "orange",
        "lightcyan",
        "mistyrose",
        "lightyellow",
        "honeydew",
        "lavender",
        "oldlace",
    ]

    def __init__(self, rates: List[Rate], monthly_fixed_charge=0):
        """
        Initialize an instance of the RateSchedule class.

        To create a schedule from JSON, see the `from_json()` and `from_json_file()` class methods.

        Parameters
        ----------
        rates: List[Rate]
            A list of Rate objects. They need to be entered in layered order so that each entry
            into the list supercedes the rate that came before. This will often follow a logic of
            Non-Peak Season Base Rate, Non-Peak Season TOU Rate, Peak Season Base Rate, Peak Season TOU Rate
        monthly_fixed_charge: float
            Fixed charge applied monthly, in USD.
        """
        super().__init__(monthly_fixed_charge)
        self.rates = rates

        self._rate_schedule = None
        self._weekend_holiday_rate_schedule = None
        self._rate_names = None

    def has_weekend_holiday_rates(self):
        """Check whether this rate schedule is different for weekends and holidays vs. weekdays."""
        return any([rate.days_applicable != DaysApplicable.ALL for rate in self.rates])

    @property
    def rate_names(self):
        """List of names for each rate."""
        if self._rate_names is None:
            self.build_rate_schedule()
        return self._rate_names

    @property
    def rate_schedule(self):
        """Getter for the rate_schedule attribute."""
        if self._rate_schedule is None:
            self.build_rate_schedule()
        return self._rate_schedule

    def rate_matrix(self, weekend_holiday=False):
        """
        Get a rate matrix with rates ($/kWh) per [month, hour].

        Parameters
        ----------
        weekend_holiday: boolean
            If true, get the weekend/holiday rate matrix. Otherwise, get the weekday matrix.
        """
        if weekend_holiday:
            rate_schedule_matrix, volumetric_rate_dict = self.weekend_holiday_rate_schedule
        else:
            rate_schedule_matrix, volumetric_rate_dict = self.rate_schedule

        # rate index for each [month, hour]
        rate_schedule_matrix = rate_schedule_matrix.astype(int)

        # vector of each rate
        volumetric_rates = np.array(
            [volumetric_rate_dict.get(k, 0.0) for k in range(max(volumetric_rate_dict.keys()) + 1)]
        )

        return volumetric_rates[rate_schedule_matrix]

    @property
    def weekend_holiday_rate_schedule(self):
        """Getter for the weekend_holiday_rate_schedule attribute."""
        if self._weekend_holiday_rate_schedule is None:
            self.build_rate_schedule(weekend_holiday=True)
        return self._weekend_holiday_rate_schedule

    def build_rate_schedule(self, weekend_holiday=False):
        """
        Build out the matrix of the rate schedule by layering all rates input to the class.

        Calculates and sets these attributes:
            rate_schedule: (np.array) a 12x24 numpy array where each entry represents the key in the rate dict
              that can be used to fetch the corresponding volumetric rate.
            rate_dict: (dict) a dictionary with keys that correspond to the entries in the rate schedule matrix
              and values that correspond to the volumetric rate for each of those rates
            rate_names: (dict) a dictionary with the same keys as rate_dict, and values that contain
              the name of each rate.

        Parameters
        ----------
        weekend_holiday: boolean
            If true, calculates and sets weekend_holiday_rate_schedule instead of rate_schedule.
        """
        # Create an empty 12x24 array to hold the results
        days_applicable = WEEKENDS_HOLIDAYS_OPTIONS if weekend_holiday else WEEKDAYS_OPTIONS

        rate_schedule = np.zeros((12, 24), dtype=int)
        rate_dict = {}
        rate_names = {}
        # Iterate through each and layer on top of the previous
        for i, rate in enumerate(self.rates):
            if rate.days_applicable in days_applicable:
                # Mask array will make sure that only replacing the values on previously entered rates where
                # applicable. For example, mask will make sure seasonal rate is only applied in the summer
                rate_schedule[rate.mask_array == rate.mask_array.max()] = i
                rate_dict[i] = rate.volumetric_rate
            rate_names[i] = rate.name

        # TODO: split these into separate variables
        if weekend_holiday:
            self._weekend_holiday_rate_schedule = (rate_schedule, rate_dict)
        else:
            self._rate_schedule = (rate_schedule, rate_dict)

        self._rate_names = rate_names

    def to_json(self) -> str:
        """Return a JSON representation of this rate schedule."""
        return json.dumps(
            {
                "rates": [asdict(rate) for rate in self.rates],
                "monthly_fixed_charge": self._monthly_fixed_charge,
            }
        )

    def to_json_file(self, filename: str):
        """
        Save the JSON representation of this rate schedule to a file.

        Parameters
        ----------
        filename: str
            The name of the file to write the rate schedule to. This will overwrite any existing file contents.
        """
        with open(filename, "w") as fp:
            fp.write(self.to_json())

    def __apply_json(self, json_dict):
        self._monthly_fixed_charge = json_dict.get("monthly_fixed_charge", 0)
        self.rates = [Rate(**rate) for rate in json_dict.get("rates")]

        # Clear cached variables
        self._rate_schedule = None
        self._weekend_holiday_rate_schedule = None
        self._rate_names = None

    @classmethod
    def from_json(cls, json_str: str):
        """Create a new RateSchedule from a JSON string that was exported from another RateSchedule."""
        d = json.loads(json_str)
        rs = cls(rates=[])
        rs.__apply_json(d)
        return rs

    @classmethod
    def from_json_file(cls, filename: str):
        """Create a new RateSchedule from a JSON file that was exported from another RateSchedule."""
        with open(filename, "r") as fp:
            d = json.load(fp)
            rs = cls(rates=[])
            rs.__apply_json(d)
            return rs

    @classmethod
    def color(cls, i):
        """Get color number i from the list of colors."""
        return cls.COLORS[int(i) % len(cls.COLORS)]

    @property
    def schedule_html(self) -> str:
        """
        Get schedule in a nice HTML table.

        Can be displayed in a notebook with display(HTML(rate.schedule_html())).
        """

        def build_rate_table(rates, rate_names):
            """Create an HTML table that lists the rate IDs and charges."""
            return (
                "<table>"
                + "".join(
                    [
                        f"<tr> <td bgcolor={self.color(i)}> {i}  </td> <td> $ {r} / kWh <br> </td>"
                        f"<td> {rate_names[i]} </td> </tr>"
                        for i, r in rates.items()
                    ]
                )
                + "</table>"
            )

        def build_schedule_table(schedule, rates):
            """Create an HTML table that shows the rate for each month and hour of the year."""

            def build_row(row, month, rates=None):
                rate_class = 'class="centered"'
                if rates is None:
                    cells = [f"<td {rate_class} bgcolor={self.color(i)}>{int(i)}</td>" for i in row]
                else:
                    cells = [f"<td {rate_class} bgcolor={self.color(i)}>{rates.get(int(i)):.2f}</td>" for i in row]
                return f"<tr> <td><strong>{month}</strong></td> {''.join(cells)} </tr>"

            hour_class = 'class="hour"'
            table_header = f"<tr> <th></th> {''.join([f'<th {hour_class}> {h} </th>' for h in HOURS[:-1]])} </tr>"
            # Use a unique ID with the buttons and tables, so if a notebook displays multiple results, the buttons don't
            # refer to the wrong table.
            id_key = uuid.uuid4().hex

            return f"""<h3> Schedule </h3>
                <script>
                    function toggle_tables(key, show_ids) {{
                        document.getElementById('rate_id_table_'+key).hidden=!show_ids;
                        document.getElementById('rate_value_table_'+key).hidden=show_ids;
                    }}
                </script>

                <input type='radio' name='table_type_{id_key}' id='rate_ids_{id_key}' onclick='toggle_tables(\"{id_key}\", true)'>
                <label for='rate_ids_{id_key}'>Rate IDs</label>
                <input type='radio' name='table_type_{id_key}' id='rate_charges_{id_key}' onclick='toggle_tables(\"{id_key}\", false)' checked>
                <label for='rate_charges_{id_key}'>Charges ($/kWh)</label>

                <table id='rate_id_table_{id_key}' hidden>
                    {table_header}
                    {"".join([build_row(r, MONTHS[m]) for m, r in enumerate(schedule)])}
                </table>

                <table id='rate_value_table_{id_key}'>
                    {table_header}
                    {"".join([build_row(r, MONTHS[m], rates) for m, r in enumerate(schedule)])}
                </table>"""

        schedule, rates = self.rate_schedule
        rates_table = f"""
            <h2>{"Weekdays" if self.has_weekend_holiday_rates() else "All days"}</h2>
            <strong>Monthly fixed charge:</strong> ${self.monthly_fixed_charge}<br><br>
            {build_rate_table(rates, self.rate_names)}
        """
        schedule_table = build_schedule_table(schedule, rates)

        weekend_holiday_schedule, weekend_holiday_rates = self.weekend_holiday_rate_schedule
        weekend_holiday_rates_table = f"""
            <h2>Weekends and Holidays</h2>
            {build_rate_table(weekend_holiday_rates, self.rate_names)}
        """
        weekend_holiday_schedule_table = build_schedule_table(weekend_holiday_schedule, weekend_holiday_rates)

        style = """<style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
            }

            td,th {
                padding: 5px;
                padding-left: 5px;
                padding-right: 5px;
            }

            td.centered {
                text-align: center;
            }

            th.hour {
                width: 2ch;
                text-align: center;
            }
        </style>"""

        full_html = style + rates_table + schedule_table

        if self.has_weekend_holiday_rates():
            full_html = full_html + weekend_holiday_rates_table + weekend_holiday_schedule_table

        return full_html
