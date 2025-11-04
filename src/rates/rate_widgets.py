"""
Widgets for setting up rate schedules in Databricks or other python notebooks.

usage:
    # Create and display the widget
    rate_picker = RateScheduleWidget()

    # In the next cell, get the resulting rate schedule.
    rate_schedule = rate_picker.get_rate_schedule()

    # To save the resulting rate schedule in your code, you can export it to JSON:
    s = rate_schedule.to_json()
    print(s)

    # And then load from JSON later:
    new_schedule = RateSchedule(rates=None)
    new_schedule.import_from_json(s)
"""

from collections import OrderedDict
from typing import Optional

import ipywidgets as widgets
from IPython.display import HTML, display

from rates.rate_structure import HOURS, MONTHS, DaysApplicable, Rate, RateSchedule

MONTH_OPTIONS = [(f"{i + 1} - {month}", i + 1) for i, month in enumerate(MONTHS)]


class RateWidget:
    """Widget for defining a single Rate."""

    def __init__(self, rate_id, delete_func, base_rate=False, rate: Optional[Rate] = None):
        """
        Widget for defining a single Rate.

        Call get_widget() to get the ipywidgets object.

        Parameters
        ----------
        rate_id: int
            ID for this rate, to identify it for deletion.
        delete_func: Callable[[widgets.Button], None]
            Function to call when delete button is clicked.
            The button instance will be passed as its single argument, and
            the button will have a rate_id attribute to identify it.
        base_rate: bool
            If true, this rate must include all months and hours, and
            can not be deleted. This is used to ensure that every time will be covered
            by at least one rate.
        rate: Optional[Rate]
            An optional Rate object. If provided, the widget will be pre-populated
            with the attributes of this Rate.
        """
        self.base_rate = base_rate
        self.id = rate_id

        if not rate:
            # If a Rate object is not provided, use the same defaults that the Rate class uses.
            rate = Rate(0)

        self.rate = widgets.BoundedFloatText(
            value=rate.volumetric_rate,
            min=0,
            description="Volumetric Rate ($/kWh):",
            style={"description_width": "auto"},
        )

        self.start_month = widgets.Dropdown(
            options=MONTH_OPTIONS,
            value=rate.seasonal_start_month,
            description="Start month:",
            style={"description_width": "auto"},
            disabled=base_rate,
        )
        self.end_month = widgets.Dropdown(
            options=MONTH_OPTIONS,
            value=rate.seasonal_end_month - 1,
            description="through end month:",
            style={"description_width": "auto"},
            disabled=base_rate,
        )

        self.start_time = widgets.Dropdown(
            options=[(f"{h:02}:00", h) for h in HOURS],
            value=rate.tou_start_hour,
            description="Start time:",
            disabled=base_rate,
        )
        self.end_time = widgets.Dropdown(
            options=[(f"{h:02}:00", h) for h in HOURS],
            value=rate.tou_end_hour,
            description="to end time:",
            disabled=base_rate,
        )

        self.name = widgets.Text(
            value=rate.name,
            placeholder="Short name or description",
            description="Name:",
        )

        self.delete_button = widgets.Button(description="Remove", disabled=base_rate)
        self.delete_button.rate_id = self.id
        self.delete_button.on_click(delete_func)

        self.days_applicable = widgets.ToggleButtons(
            description="Days this rate applies:",
            options=[d.value for d in DaysApplicable],
            value=rate.days_applicable,
            style={"description_width": "initial", "button_width": "auto"},
            button_style="info",
            disabled=base_rate,
        )

    def get_widget(self, index=0):
        """
        Get a widget that contains all the input fields and buttons for this rate.

        Parameters
        ----------
        index: int
            The index of this rate in the list of rates. Used to give it the right color.
        """
        return widgets.VBox(
            [
                widgets.HTML("<strong>Base rate</strong>" if self.base_rate else None),
                widgets.HBox(
                    [
                        # Preview the color assigned to this rate
                        widgets.HTML(
                            f"<div style='background:{RateSchedule.color(index)};"
                            f"width:25px;height:25px;text-align:center;vertical-align:middle'>{index}<div>"
                        ),
                        widgets.VBox([self.name, self.start_month, self.end_month]),
                        widgets.VBox([self.rate, self.start_time, self.end_time]),
                    ]
                ),
                self.days_applicable,
                self.delete_button,
            ],
            layout=widgets.Layout(border="solid", margin="3px", padding="6px"),
        )


class RateScheduleWidget:
    """
    Widget for defining a RateSchedule.

    Call get_rate_schedule() to get the resulting RateSchedule.
    """

    def __init__(self, rate_schedule: Optional[RateSchedule] = None):
        """
        Widget for defining a rate schedule.

        Parameters
        ----------
        rate_schedule: Optional[RateSchedule]
            An optional RateSchedule object. If provided, the widget will
            be pre-populated with the attributes of this rate schedule.
        """
        # ID to use for the next rate widget created
        self.next_rate_id = 0
        # Map from rate ID to rate widget.
        self.rates = OrderedDict()

        if rate_schedule:
            for i, rate in enumerate(rate_schedule.rates):
                self.__add_rate(base_rate=(i == 0), rate=rate)
        else:
            self.__add_rate(base_rate=True)

        self.fixed_rate = widgets.FloatText(
            value=rate_schedule.monthly_fixed_charge if rate_schedule else 0,
            description="Fixed Monthly Rate ($):",
            style={"description_width": "auto"},
        )

        self.new_rate_button = widgets.Button(description="Add new rate")

        def on_new_rate_button_click(b):
            self.__add_rate()
            self.__update()

        self.new_rate_button.on_click(on_new_rate_button_click)

        # Button to print out the rate structure to self.output
        self.display_button = widgets.Button(description="Display")
        self.output = widgets.Output()

        def on_display_button_clicked(b):
            self.output.clear_output(wait=True)
            with self.output:
                display(HTML(self.get_rate_schedule().schedule_html))
            display(self.output)

        self.display_button.on_click(on_display_button_clicked)

        self.big_box = widgets.VBox([])
        self.__update()
        self.display()

    def __get_on_delete_fn(self):
        """Get function to run when a delete button is clicked."""

        def on_delete_fn(b):
            del self.rates[b.rate_id]
            self.__update()

        return on_delete_fn

    def __update(self):
        """
        Update the widget to reflect the current list of rates.

        Should be called after a rate is added or removed.
        """
        self.big_box.children = (
            [self.fixed_rate]
            + [p.get_widget(i) for i, p in enumerate(self.rates.values())]
            + [self.new_rate_button, self.display_button, self.output]
        )

    def __add_rate(self, base_rate=False, rate: Rate = None):
        """
        Add a new rate widget.

        Parameters
        ----------
        base_rate: bool
            Whether the rate widget should be for a base rate that covers all
            days and times.
        rate: Optional[Rate]
            An optional Rate object. If provided, the new rate widget will be
            pre-populated with the attributes of this Rate.
        """
        p = RateWidget(self.next_rate_id, self.__get_on_delete_fn(), base_rate, rate)
        self.rates[p.id] = p
        self.next_rate_id += 1

    def display(self):
        """Display the widget."""
        display(self.big_box)

    def get_rate_schedule(self):
        """Get RateSchedule object."""
        rate_list = []
        for widget in self.rates.values():
            rate_list.append(
                Rate(
                    name=widget.name.value,
                    volumetric_rate=widget.rate.value,
                    tou_start_hour=widget.start_time.value,
                    tou_end_hour=widget.end_time.value,
                    seasonal_start_month=widget.start_month.value,
                    seasonal_end_month=widget.end_month.value + 1,
                    days_applicable=widget.days_applicable.value,
                )
            )
        return RateSchedule(rates=rate_list, monthly_fixed_charge=self.fixed_rate.value)
