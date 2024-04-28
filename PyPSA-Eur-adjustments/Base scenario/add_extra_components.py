# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config/config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from add_electricity import load_costs, sanitize_carriers

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def attach_storageunits(n, costs, extendable_carriers, max_hours):
    carriers = extendable_carriers["StorageUnit"]

    n.madd("Carrier", carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        roundtrip_correction = 0.5 if carrier == "battery" else 1

        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_store[carrier], "efficiency"]
            ** roundtrip_correction,
            efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"]
            ** roundtrip_correction,
            max_hours=max_hours[carrier],
            cyclic_state_of_charge=True,
        )

# changed by Shufen
def attach_stores(n, costs, extendable_carriers):
    carriers = extendable_carriers["Store"]

    n.madd("Carrier", carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Electrolysis",
            bus0=buses_i,
            bus1=h2_buses_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Fuel Cell",
            bus0=h2_buses_i,
            bus1=buses_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            # NB: fixed cost is per MWel
            capital_cost=costs.at["fuel cell", "capital_cost"]
            * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
        )

    if "battery" in carriers:
        b_buses_i = n.madd(
            "Bus", buses_i + " battery", carrier="battery", **bus_sub_dict
        )

        n.madd(
            "Store",
            b_buses_i,
            bus=b_buses_i,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["battery storage", "capital_cost"],
            marginal_cost=costs.at["battery", "marginal_cost"],
        )

        n.madd("Carrier", ["battery charger", "battery discharger"])

        n.madd(
            "Link",
            b_buses_i + " charger",
            bus0=buses_i,
            bus1=b_buses_i,
            carrier="battery charger",
            # the efficiencies are "round trip efficiencies"
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

        n.madd(
            "Link",
            b_buses_i + " discharger",
            bus0=b_buses_i,
            bus1=buses_i,
            carrier="battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

    if "MDES" in carriers:
        md_buses_i = n.madd("Bus", buses_i + " MDES", carrier="MDES", **bus_sub_dict)

        # energy cost=1€/kWh need to be converted into capital cost = 1*1000*0.081[€/MWh], annuity(7.1%,30)=0.081
        n.madd(
            "Store",
            md_buses_i,
            bus= md_buses_i,
            carrier="MDES",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost= 81, 
        )

        n.madd("Carrier", ["mdes-charger", "mdes-discharger"])

        # the weighted power capacity cost is 100 €/kWh but need to be separated into two parts (charger and discharger)
        n.madd(
            "Link",
            md_buses_i + " Charger",
            bus0=buses_i,
            bus1=md_buses_i,
            carrier="mdes-charger",
            p_nom_extendable=True,
            efficiency= 0.7 **0.5,
            capital_cost= 8100/2, # 100*1000*0.081= €/MWh
            marginal_cost= 0,
        )

        n.madd(
            "Link",
            md_buses_i + " Discharger",
            bus0=md_buses_i,
            bus1=buses_i,
            carrier="mdes-discharger",
            p_nom_extendable=True,
            capital_cost= 8100/2, 
            efficiency= 0.7 **0.5,
            marginal_cost= 0,
        )

# for sensitivity analysis
    if "CAES" in carriers:
        caes_buses_i = n.madd("Bus", buses_i + " CAES", carrier="CAES", **bus_sub_dict)

        n.madd(
            "Store",
            caes_buses_i,
            bus=caes_buses_i,
            carrier="CAES",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=405, 
        )

        n.madd("Carrier", ["CAES charger", "CAES discharger"])

        n.madd(
            "Link",
            caes_buses_i + " charger",
            bus0=buses_i,
            bus1=caes_buses_i,
            carrier="CAES charger",
            p_nom_extendable=True,
            efficiency=0.7 **0.5, # costs.at["Compressed-Air-Adiabatic-bicharger", "efficiency"]
            capital_cost=24300, # costs.at["Compressed-Air-Adiabatic-bicharger", "capital_cost"]
           
        )

        n.madd(
            "Link",
            caes_buses_i + " discharger",
            bus0=caes_buses_i,
            bus1=buses_i,
            carrier="CAES discharger",
            p_nom_extendable=True,
            efficiency=0.7 **0.5, # costs.at["Compressed-Air-Adiabatic-bicharger", "efficiency"] ** 0.5
        
        )

    if "PTES" in carriers:
        ptes_buses_i = n.madd("Bus", buses_i + " PTES", carrier="PTES", **bus_sub_dict)

        n.madd(
            "Store",
            ptes_buses_i,
            bus=ptes_buses_i,
            carrier="PTES",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=405, # costs.at["Pumped-Heat-store", "capital_cost"]
        )

        n.madd("Carrier", ["PTES charger", "PTES discharger"])

        n.madd(
            "Link",
            ptes_buses_i + " charger",
            bus0=buses_i,
            bus1=ptes_buses_i,
            carrier="PTES charger",
            p_nom_extendable=True,
            efficiency=0.7 **0.5, # costs.at["Pumped-Heat-charger", "efficiency"]
            capital_cost=44550, # costs.at["Pumped-Heat-charger", "capital_cost"]
            
        )

        n.madd(
            "Link",
            ptes_buses_i + " discharger",
            bus0=ptes_buses_i,
            bus1=buses_i,
            carrier="PTES discharger",
            p_nom_extendable=True,
            efficiency=0.7 **0.5, #costs.at["Pumped-Heat-discharger", "efficiency"]
            capital_cost=44550, # costs.at["Pumped-Heat-discharger", "capital_cost"]
            
        )
    
    if "LAES" in carriers:
        laes_buses_i = n.madd("Bus", buses_i + " LAES", carrier="LAES", **bus_sub_dict)

        n.madd(
            "Store",
            laes_buses_i,
            bus=laes_buses_i,
            carrier="LAES",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["Liquid-Air-store", "capital_cost"],
        )

        n.madd("Carrier", ["LAES charger", "LAES discharger"])

        n.madd(
            "Link",
            laes_buses_i + " charger",
            bus0=buses_i,
            bus1=laes_buses_i,
            carrier="LAES charger",
            p_nom_extendable=True,
            efficiency=costs.at["Liquid-Air-charger", "efficiency"],
            capital_cost=costs.at["Liquid-Air-charger", "capital_cost"],
            
        )

        n.madd(
            "Link",
            laes_buses_i + " discharger",
            bus0=laes_buses_i,
            bus1=buses_i,
            carrier="LAES discharger",
            p_nom_extendable=True,
            efficiency=costs.at["Liquid-Air-discharger", "efficiency"],
            capital_cost=costs.at["Liquid-Air-discharger", "capital_cost"],
            
        )

    if "VRFB" in carriers:
        vrfb_buses_i = n.madd("Bus", buses_i + " VRFB", carrier="VRFB", **bus_sub_dict)

        n.madd(
            "Store",
            vrfb_buses_i,
            bus=vrfb_buses_i,
            carrier="VRFB",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=4050, # costs.at["Vanadium-Redox-Flow-store", "capital_cost"]
        )

        n.madd("Carrier", ["VRFB charger", "VRFB discharger"])

        n.madd(
            "Link",
            vrfb_buses_i + " charger",
            bus0=buses_i,
            bus1=vrfb_buses_i,
            carrier="VRFB charger",
            p_nom_extendable=True,
            efficiency=0.8 **0.5, # costs.at["Vanadium-Redox-Flow-bicharger", "efficiency"]
            capital_cost=8100, # costs.at["Vanadium-Redox-Flow-bicharger", "capital_cost"]
    
        )

        n.madd(
            "Link",
            vrfb_buses_i + " discharger",
            bus0=vrfb_buses_i,
            bus1=buses_i,
            carrier="VRFB discharger",
            p_nom_extendable=True,
            efficiency=0.8 **0.5, # costs.at["Vanadium-Redox-Flow-bicharger", "efficiency"] ** 0.5
        )


def attach_hydrogen_pipelines(n, costs, extendable_carriers):
    as_stores = extendable_carriers.get("Store", [])

    if "H2 pipeline" not in extendable_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]]
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.add("Carrier", "H2 pipeline")

    n.madd(
        "Link",
        h2_links.index,
        bus0=h2_links.bus0.values + " H2",
        bus1=h2_links.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * h2_links.length,
        efficiency=costs.at["H2 pipeline", "efficiency"],
        carrier="H2 pipeline",
    )




if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_extra_components", simpl="", clusters=5)
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    extendable_carriers = snakemake.params.extendable_carriers
    max_hours = snakemake.params.max_hours

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs, snakemake.params.costs, max_hours, Nyears
    )

    attach_storageunits(n, costs, extendable_carriers, max_hours)
    attach_stores(n, costs, extendable_carriers)
    attach_hydrogen_pipelines(n, costs, extendable_carriers)


    sanitize_carriers(n, snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
