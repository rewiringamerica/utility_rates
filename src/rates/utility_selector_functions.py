"""
Widgets and functions for easily finding counties served by a particular electric utility in Databricks or other Python notebooks.

NOTE: These results represent utility service areas at county resolution, which leads to two important caveats:
1. The same county may appear in multiple utility service areas
2. When used to identify ResStock buildings in a utility territory, this provides an upper-bound estimate
   as some households in boundary counties or municipal enclaves may be served by neighboring utilities.

Usage:
    # Create and display the dropdown widget
    selector = create_electric_utility_widget()
    # -- User interacts with widgets in notebook to select state and utility -- #

    # In the next cell, retrieve a list of county GEOIDs in the selected state and utility
    county_geoids = select_counties_in_utility(selector)
"""

from typing import Dict, List

import ipywidgets as widgets
import pyspark.sql.functions as F
from databricks.sdk.runtime import *
from IPython.display import display


def create_electric_utility_widget(
    state_default: str = "-- Select State --", utility_default: str = "-- Select Utility --"
) -> Dict[str, str]:
    """
    Create an interactive state and electric utility selector widget to make it easier to find the name of a utility.

    The output will then generally get passed directly into select_county_geoids_in_utility.

    This function loads utility data from a Spark table from EIA Form 861 that maps uti
    and presents two dropdown menus populated from that table.
    - State selector
    - Utility selector

    Parameters
    ----------
    state_default : str
        State that should initially be selected when dropdown is initialized. Default is "-- Select State --".
    utility_default : str
        Utility that should initially be selected when dropdown is initialized. Default is "-- Select Utility"

    Returns
    -------
    A dictionary containing the current selections with keys:
        - 'state': Currently selected state (str)
        - 'utility': Currently selected utility (str or None if not selected)

    Example:
        >>> selector = create_electric_utility_widget()
        >>> # Interact with widgets in notebook
        >>> print(selector['state'], selector['utility'])  # Shows current selections
    """
    state_init_value = "-- Select State --"
    utility_init_value = "-- Select Utility --"

    # Load table of [state, utility_name]
    df = spark.table("geographic_crosswalk.electric_utility_to_county_geoid")
    state_utility_df = df[["state", "utility_name"]].distinct().toPandas()

    # Initialize widgets
    state_dropdown = widgets.Dropdown(
        options=[state_default] + sorted(state_utility_df["state"].unique()),
        value=state_default,
        description="State:",
    )

    utility_dropdown = widgets.Dropdown(options=[utility_default], value=utility_default, description="Utility:")

    # Store selections
    utility_selection = None if utility_default == "-- Select Utility --" else utility_default
    selections = {"state": state_dropdown.value, "utility": utility_selection}

    # Populate utility widget
    def populate_utilities(state: str):
        """
        Populate the utility dropdown based on the selected state.

        Parameters
        ----------
        state: str
            The selected state to filter utilities by
        """
        utilities = sorted(state_utility_df[state_utility_df["state"] == state]["utility_name"].unique())
        utility_dropdown.options = [utility_default] + utilities

    # Update utilities when state changes
    def on_state_change(change: Dict[str, str]):
        """
        Handle state dropdown changes by updating utilities and resetting selections.

        Parameters
        ----------
        change : dict
            The widget change event containing:
                        - 'new': The new state value
                        - 'old': The previous state value
        """
        selections["state"] = change["new"]
        populate_utilities(change["new"])
        utility_dropdown.value = utility_default
        selections["utility"] = None

    # track utility selection
    def on_utility_change(change: Dict[str, str]):
        """
        Tracks utility selection changes in the selections dictionary.

        Parameters
        ----------
        change: dict
            The widget change event containing:
                        - 'new': The new utility value
                        - 'old': The previous utility value
        """
        if change["new"] != utility_default:
            selections["utility"] = change["new"]

    # Set up observers
    state_dropdown.observe(on_state_change, names="value")
    utility_dropdown.observe(on_utility_change, names="value")

    # Initialize
    populate_utilities(state_dropdown.value)

    # Display widgets
    display(widgets.VBox([state_dropdown, utility_dropdown]))

    return selections


def select_county_geoids_in_utility(selector: Dict[str, str]) -> List[str]:
    """
    Retrieve county GEOIDs for the currently selected state and utility and prints names of counties.

    Parameters
    ----------
    selector: dict
        The selection dictionary from create_utility_selector() with keys:
            - 'state': Selected state (str)
            - 'utility': Selected utility (str)

    Returns
    -------
        list: A list of county GEOIDs (str) for the selected utility area

    Example:
        >>> county_geoids = select_counties_in_utility(selector)
    """
    # select county names and geoids in the electric utility x state and convert to pandas
    selected_counties_df = (
        spark.table("geographic_crosswalk.electric_utility_to_county_geoid")
        .where(F.col("state") == selector["state"])
        .where(F.col("utility_name") == selector["utility"])
        .select("county_name", "county_geoid")
        .toPandas()
    )

    # print matching county names for visual inspection
    print(f"Counties in {selector['utility']}: {sorted(selected_counties_df.county_name)}")
    # return matching county geoids
    return selected_counties_df.county_geoid.to_list()
