# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates plots for optimised power network topologies and regional generation,
storage and conversion capacities built.
"""

import logging

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import configure_logging
from plot_summary import preferred_order, rename_techs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from plot_power_network_clustered import load_projection
logger = logging.getLogger(__name__)


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["NH3", "Haber-Bosch", "ammonia cracker", "ammonia store"]:
        return "ammonia"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    # elif "solar" in tech:
    #     return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


#def assign_location(n):
    #for c in n.iterate_components(n.one_port_components | n.branch_components):
        #ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        #for i in ifind.value_counts().index:
            # these have already been assigned defaults
            #if i == -1:
                #continue
            #names = ifind.index[ifind == i]
            #c.df.loc[names, "location"] = names.str[:i]
def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.unique():
            names = ifind.index[ifind == i]
            if i == -1:
                c.df.loc[names, "location"] = ""
            else:
                c.df.loc[names, "location"] = names.str[:i]            

def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def plot_map(
    n,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=2e10,
    transmission=True, # default=False
    with_legend=True,
):
    tech_colors = snakemake.params.plotting["tech_colors"]

    assign_location(n)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)
        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = (
            (df_c.capital_cost * df_c[attr])
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )

        costs = pd.concat([costs, costs_c], axis=1)

        logger.debug(f"{comp}, {costs}")

    costs = costs.groupby(costs.columns, axis=1).sum()

    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    costs = costs.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = snakemake.params.plotting.get("eu_node_location", dict(x=-5.5, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    threshold = 100e6  # 100 mEUR/a
    carriers = costs.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)
    


    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0 #default=500.0
    line_upper_threshold = 1e4 #default=1e4
    linewidth_factor = 4e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    title = "added grid"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        # line_widths_opt = n.lines.s_nom_opt
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)
    # line_widths_opt = line_widths_opt.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    sizes = [20, 10, 5, 1] # [20, 10, 5]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]
    
    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.10),
        labelspacing=1.2,
        frameon=False,
        handletextpad=0,
        title="system cost",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5, 2] # [10,5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.10),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    legend_kw = dict(
        bbox_to_anchor=(1.60, 1.10),
        frameon=False,
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches( 
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )

    fig.savefig(snakemake.output.map, bbox_inches="tight")


def plot_dispatch_map(
    n,
    components=["links", "storage_units", "generators"],
    bus_size_factor=2e5,
    transmission=True, # default=False
    with_legend=True,
):
    tech_colors = snakemake.params.plotting["tech_colors"]

    assign_location(n)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    storage = n.stores.query("carrier == 'MDES'").e_nom_opt
    new_index = [idx.split(' ')[0] + ' ' + idx.split(' ')[1] for idx in storage.index]
    storage.index = new_index
    # merge the regions and capacity
    nodes = regions
    epsg = 3035
    nodes["Area"] = nodes.to_crs(epsg=epsg).area.div(1e6)
    nodes["MDES"] = storage.div(1e6) #TWh
    nodes["MDES"] = nodes["MDES"].where(nodes["MDES"]>=0.0004) # storage threshold for MDES
    print(nodes)
    nodes = nodes.to_crs(proj.proj4_init)


    power = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_p = getattr(n, comp)
        #print(df_p)
        if df_p.empty:
            continue

        df_p["nice_group"] = df_p.carrier.map(rename_techs_tyndp)

        power_p = (
            df_p["p_nom_opt"].groupby([df_p.location, df_p.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )

        power = pd.concat([power, power_p], axis=1)
        #print(power)
        logger.debug(f"{comp}, {power}")

    power = power.groupby(power.columns, axis=1).sum()
    power.drop(list(power.columns[(power == 0.0).all()]), axis=1, inplace=True)
    new_columns = preferred_order.intersection(power.columns).append(
        power.columns.difference(preferred_order)
    )
    power = power[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    power = power.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = snakemake.params.plotting.get("eu_node_location", dict(x=-5.5, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = power.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        power.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    power.index = pd.MultiIndex.from_tuples(power.index.values)

    threshold = 0
    carriers = power.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)



    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0 #default=500.0
    line_upper_threshold = 1e4 #default=1e4
    linewidth_factor = 4e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    title = "added grid"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        # line_widths_opt = n.lines.s_nom_opt
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)
    # line_widths_opt = line_widths_opt.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(10, 10) #(10,11) for base case

    n.plot(
        bus_sizes=power / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    regions_plot=nodes.plot(
        ax=ax,
        column="MDES",
        cmap="Oranges",
        linewidths=0,
        legend=True,
        vmax=5, # can be scaled
        vmin=0,
        legend_kwds={
        "label": r"MDES storage capacity [TWh]",
        "shrink": 0.6,
        "extend": "max",
        "orientation":"horizontal",
        "anchor":(0.5, 2.2),
        },
    )

    sizes = [300, 150, 50, 10] #[200, 100, 50, 10]
    labels = [f"{s} GW" for s in sizes] #GW
    sizes = [s / bus_size_factor*1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.00), # (-0.15, 1.00) for base case
        labelspacing=1.5,
        frameon=False,
        handletextpad=0,
        title="Generation",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [15, 10, 5, 2] # [10,5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.10, 1.00), # (0.10, 1.00) for base case
        frameon=False,
        labelspacing=1.2,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    legend_kw = dict(
        ncol=3, # ncol=4 for base case
        bbox_to_anchor=(1.0, -0.12), # (1.05, -0.12) for base case
        frameon=False,
        #title="Technology", 
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches( 
            ax,
            colors,
            labels,
            legend_kw=legend_kw, 
        )

    fig.savefig(
        snakemake.output.dispatch, bbox_inches="tight")


def plot_storage_map(
    n,
    components=["stores"], #components=["stores", "storage_units"]
    bus_size_factor=6e6,
    transmission=True, # default=False
    with_legend=True,
):
    tech_colors = snakemake.params.plotting["tech_colors"]

    assign_location(n)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    energy = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_e = getattr(n, comp)
        
        if df_e.empty:
            continue

        df_e["nice_group"] = df_e.carrier.map(rename_techs_tyndp)

        # remove the hydro storage and only display the added storage technologies
        df_e = df_e.loc[~(df_e["nice_group"] == "hydroelectricity")]
        
        #attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        energy_stores = pd.DataFrame()
        energy_units = pd.DataFrame()

        if comp == "stores":
            energy_stores = (
                df_e["e_nom_opt"].groupby([df_e.location, df_e.nice_group])
                .sum()
                .unstack()
                .fillna(0.0)
            )
            #print(energy_stores)
        elif comp == "storage_units":
            energy_units = (
                (df_e["p_nom_opt"]*df_e["max_hours"]).groupby([df_e.location, df_e.nice_group])
                .sum()
                .unstack()
                .fillna(0.0)
            )
            
        # energy = pd.concat([energy, energy_stores, energy_units], axis=1)
        energy = pd.concat([energy, energy_stores], axis=1)
        logger.debug(f"{comp}, {energy}")

    energy = energy.groupby(energy.columns, axis=1).sum()

    energy.drop(list(energy.columns[(energy == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(energy.columns).append(
        energy.columns.difference(preferred_order)
    )
    energy = energy[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    energy = energy.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = snakemake.params.plotting.get("eu_node_location", dict(x=-5.5, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = energy.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        energy.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    energy.index = pd.MultiIndex.from_tuples(energy.index.values)

    threshold = 0
    carriers = energy.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)



    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0 #default=500.0
    line_upper_threshold = 1e4 #default=1e4
    linewidth_factor = 4e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    title = "added grid"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        # line_widths_opt = n.lines.s_nom_opt
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)
    # line_widths_opt = line_widths_opt.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    n.plot(
        bus_sizes=energy / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    sizes = [10, 5, 1]
    labels = [f"{s} TWh" for s in sizes]
    sizes = [s / bus_size_factor*1e6 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.10),
        labelspacing=1.2,
        frameon=False,
        handletextpad=0,
        title="Storage potential",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5, 2] # [10,5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.10),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    legend_kw = dict(
        bbox_to_anchor=(1.60, 1.10),
        frameon=False,
        title="Technology", 
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches( 
            ax,
            colors,
            labels,
            legend_kw=legend_kw, 
        )

    fig.savefig(
        snakemake.output.storage, bbox_inches="tight")

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_power_network",
            simpl="",
            opts="",
            clusters="37",
            ll="v1.0",
            # sector_opts="4380H-T-H-B-I-A-dist1",
        )

    configure_logging(snakemake)

    n1 = pypsa.Network(snakemake.input.network)
    n2 = pypsa.Network(snakemake.input.network)
    n3 = pypsa.Network(snakemake.input.network)
    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = snakemake.params.plotting["map"]

    if map_opts["boundaries"] is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    proj = load_projection(snakemake.params.plotting)

    plot_dispatch_map(n1)
    plot_map(n2)
    plot_storage_map(n3)
    