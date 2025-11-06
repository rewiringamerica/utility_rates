# rates
Modeling rate structures applied to ResStock buildings.

This repo has code to help you:
- Interactively define a rate schedule, including seasonal and TOU rates.
- Apply that rate structure to a set of homes from [ResStock](https://resstock.nrel.gov/) to calculate
  their utility bills, in either a baseline or upgrade scenario.


# Running on Databricks

## Install dependencies
This is designed to be run on Databricks clusters with the 15.4 LTS runtime (Python 3.11), though it can also
run in a local Jupyter notebook.


## Examples
We recommend starting with the [`basic_demo.ipynb`](notebooks/basic_demo/basic_demo.ipynb). You can use this notebook directly
to enter a rate schedule and do some basic analysis, or copy from it to create your own.
It provides an example of [the required data input format](notebooks/basic_demo/resstock_electric_load_profile_baseline_sample.csv). 

There are also 4 additional notebooks in the `notebooks` folder that demonstrate more advanced real world examples. Note that these depend
on load profile data stored outside this repository.

* [`duquesne_heating_rate_hp.ipynb`](notebooks/duquesne_heating_rate_hp/duquesne_heating_rate_hp.ipynb): Heat pump savings for single family occupied homes in the Duquesne Light Territory with the flat default residential rate and the seasonal electric heating rate. This is a demonstration simple rate case for the RATES module: applying rate structures to baseline and/or upgrade load profiles and calculating total bill savings over all fuels under the change in rate and/or appliances. *This is also also provides a good template of a simple rate analysis, highlighting cells where updates should be made, and providing many example plots.*
* [`san_jose_hp_hpwh.ipynb`](notebooks/san_jose_hp_hpwh/san_jose_hp_hpwh.ipynb): Heat pump and heat pump water heater savings for San Jose under various PG&E TOU rates. This is an example of a more complex analysis involving comparing multiple time of use rates, baseline allowances, and multiple upgrades. 
* [`duke_capacity_payments.ipynb`](notebooks/duke_capacity_payments/duke_capacity_payments.ipynb): Capacity payments for homes installing a heat pump/heat pump water heater in Duke's North Carolina's utility territory under their capacity payment structure, which is meant to incentivise peak reduction. This is very similar to a time of use rate, except that one is paid for using less electricity at certain times due to an efficiency upgrade. This demonstrates an alternate use case, where we apply a rate structure directly the savings load profile (the difference between upgrade and baseline) to calculate the total payments to the customer.
* [`black_hills_hourly_emissions.ipynb`](notebooks/black_hills_hourly_emissions.ipynb): Emissions reductions from gas homes in the Black Hills, CO utility territory installing a heat pump while accounting for the time-varying emissivity of the grid. Since the emissions/kWh of electricity varies by season and time of day depending on the grid mix, it is more accurate to calculate emissions savings using time-varying emissions factors, particularly when the increased electricity due to an upgrade will be concentrated at certain times (e.g, increased demand on winter mornings when fuel switching to a heat pump). This demonstrates an alternate use case, where we apply a emissions structure, that is kgCO2e/kWh for each (month, hour), to a baseline and upgraded electricity load profile to calculate emissions savings under the upgrade. Importantly, unlike with the first two use cases, the user does not need to provide these emissions structure since we have access to this data for each state.


# Running locally

## Install dependencies
Install dependencies and create a virtual env with [poetry](https://python-poetry.org/docs/):

```shell
poetry install
```

## Developing locally
To also install lint checks and other dev tools, install with:
```shell
poetry install --with dev
```
or add them later with
```shell
poetry sync --with dev
```

Enable [pre-commit checks](https://pre-commit.com/) with:
```shell
pre-commit install
```
# Contact
For questions, updates, or to join our open source community, please reach out to `opensource@rewiringamerica.org`. 
