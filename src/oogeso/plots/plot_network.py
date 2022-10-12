import pydot


def plot_network(
    simulator,
    timestep=None,
    filename=None,
    prog="dot",
    only_carrier=None,
    rank_direction="LR",
    plot_device_name=False,
    number_format="{:.2g}",
    hide_losses=False,
    hide_edge_label=False,
    **kwargs,
):
    """Plot energy network

    simulator : Simulation object
    timestep : int
        which timestep to show values for
    filename : string
        Name of file
    only_carrier : str or list
        Restrict energy carriers to these types (None=plot all)
    rankdir : str
        Plotting direction TB=top to bottom, LR=left to right
    numberformat : str
        specify how numbers should be represented in plot
    hide_losses : bool
        Don't show losses on edges (if any)
    hide_edgelabel : bool
        Don't show any labels on edges
    **kwargs :
        Additional arguments passed on to pydot.Dot(...)
    """

    # Idea: in general, there are "in" and "out" terminals. If there are
    # no serial devices, then these are merged into a single terminal
    # (prettier plot"). Whether the single terminal is shown as an in or out
    # terminal (left or irght), depends on whether it is an input or output
    # of a majority of the connected devices.

    optimiser = simulator.optimiser
    model = optimiser
    res = simulator.result_object

    col = {
        "t": {
            "el": "red",
            "gas": "orange",
            "heat": "darkgreen",
            "wellstream": "brown",
            "oil": "black",
            "water": "blue4",
            "hydrogen": "deepskyblue2",
        },
        "e": {
            "el": "red",
            "gas": "orange",
            "heat": "darkgreen",
            "wellstream": "brown",
            "oil": "black",
            "water": "blue4",
            "hydrogen": "deepskyblue2",
        },
        "d": "white",
        "cluster": "lightgray",
    }
    # dotG = pydot.Dot(graph_type='digraph') #rankdir='LR',newrank='false')
    dotG = pydot.Dot(graph_type="digraph", rankdir=rank_direction, **kwargs)
    if only_carrier is None:
        carriers = model.setCarrier
    elif type(only_carrier) is str:
        carriers = [only_carrier]
    else:
        carriers = only_carrier

    # plot all node and terminals:
    for n_id, node_obj in optimiser.all_nodes.items():
        cluster = pydot.Cluster(graph_name=n_id, label=n_id, style="filled", color=col["cluster"])
        terms_in = pydot.Subgraph(rank="min")
        gr_devices = pydot.Subgraph(rank="same")
        terms_out = pydot.Subgraph(rank="max")
        for carrier in carriers:
            # add only terminals that are connected to something (device or edge)
            if node_obj.is_non_trivial(carrier):
                devs = node_obj.devices
                num_in = 0
                num_out = 0
                for d, dev_obj in devs.items():
                    dev_model = dev_obj.dev_data.model
                    devlabel = "{}\n{}".format(d, dev_model)
                    if plot_device_name:
                        dev_name = dev_obj.dev_data.name
                        devlabel = "{} {}".format(devlabel, dev_name)
                    carriers_in = dev_obj.carrier_in
                    carriers_out = dev_obj.carrier_out
                    carriers_in_lim = list(set(carriers_in) & set(carriers))
                    carriers_out_lim = list(set(carriers_out) & set(carriers))
                    if (carriers_in_lim != []) or (carriers_out_lim != []):
                        gr_devices.add_node(pydot.Node(d, color=col["d"], style="filled", label=devlabel))
                    if carrier in carriers_in_lim:
                        num_in += 1
                        if timestep is None:
                            devedgelabel = ""
                        else:
                            f_in = res.device_flow[(d, carrier, "in", timestep)]
                            devedgelabel = number_format.format(f_in)
                        if carrier in node_obj.devices_serial:
                            n_in = n_id + "_" + carrier + "_in"
                        else:
                            n_in = n_id + "_" + carrier
                        dotG.add_edge(
                            pydot.Edge(
                                dst=d,
                                src=n_in,
                                color=col["e"][carrier],
                                fontcolor=col["e"][carrier],
                                label=devedgelabel,
                            )
                        )
                    if carrier in carriers_out_lim:
                        num_out += 1
                        if timestep is None:
                            devedgelabel = ""
                        else:
                            f_out = res.device_flow[(d, carrier, "out", timestep)]
                            devedgelabel = number_format.format(f_out)
                        if carrier in node_obj.devices_serial:
                            n_out = n_id + "_" + carrier + "_out"
                        else:
                            n_out = n_id + "_" + carrier
                        dotG.add_edge(
                            pydot.Edge(
                                dst=n_out,
                                src=d,
                                color=col["e"][carrier],
                                fontcolor=col["e"][carrier],
                                label=devedgelabel,
                            )
                        )

                # add in/out terminals
                label_in = carrier  # + "_in "
                label_out = carrier  # + supp + " "
                if timestep is None:
                    pass
                elif carrier in ["gas", "wellstream", "oil", "water"]:
                    label_in += "_" + number_format.format(res.terminal_pressure[(n_id, carrier, "in", timestep)])
                    label_out += "_" + number_format.format(res.terminal_pressure[(n_id, carrier, "out", timestep)])
                elif carrier == "el":
                    if optimiser.all_networks["el"].carrier_data.powerflow_method == "dc-pf":
                        label_in += "_" + number_format.format(res.el_voltage_angle[(n_id, timestep)])
                        label_out += "_" + number_format.format(res.el_voltage_angle[(n_id, timestep)])
                # Add two terminals if there are serial devices, otherwise one:
                if carrier in node_obj.devices_serial:
                    terms_in.add_node(
                        pydot.Node(
                            name=n_id + "_" + carrier + "_in",
                            color=col["t"][carrier],
                            label=label_in,
                            shape="box",
                        )
                    )
                    terms_out.add_node(
                        pydot.Node(
                            name=n_id + "_" + carrier + "_out",
                            color=col["t"][carrier],
                            label=label_out,
                            shape="box",
                        )
                    )
                else:
                    if num_out > num_in:
                        terms_out.add_node(
                            pydot.Node(
                                name=n_id + "_" + carrier,
                                color=col["t"][carrier],
                                label=label_out,
                                shape="box",
                            )
                        )
                    else:
                        terms_in.add_node(
                            pydot.Node(
                                name=n_id + "_" + carrier,
                                color=col["t"][carrier],
                                label=label_out,
                                shape="box",
                            )
                        )

        cluster.add_subgraph(terms_in)
        cluster.add_subgraph(gr_devices)
        cluster.add_subgraph(terms_out)
        dotG.add_subgraph(cluster)

    # plot all edges (per carrier):
    for carrier in carriers:
        for i, edge_obj in optimiser.all_edges.items():
            edge_data = edge_obj.edge_data
            if edge_data.carrier == carrier:
                headlabel = ""
                taillabel = ""
                if hide_edge_label:
                    edgelabel = ""
                elif timestep is None:
                    edgelabel = ""
                    if (not hide_edge_label) and hasattr(edge_data, "pressure_from"):
                        edgelabel = "{} {}-{}".format(
                            edgelabel,
                            edge_data.pressure_from,
                            edge_data.pressure_to,
                        )
                else:
                    edgelabel = number_format.format(res.edge_flow[(i, timestep)])
                    # Add loss
                    if (not hide_losses) and (res.edge_loss is not None) and ((i, timestep) in res.edge_loss):
                        losslabel = number_format.format(res.edge_loss[(i, timestep)])
                        edgelabel = "{} [{}]".format(edgelabel, losslabel)
                n_from = edge_data.node_from
                n_to = edge_data.node_to
                n_from_obj = optimiser.all_nodes[n_from]
                n_to_obj = optimiser.all_nodes[n_to]
                # name of terminal depends on whether it serial or single
                if carrier in n_from_obj.devices_serial:
                    t_out = n_from + "_" + carrier + "_out"
                else:
                    t_out = n_from + "_" + carrier
                if carrier in n_to_obj.devices_serial:
                    t_in = n_to + "_" + carrier + "_in"
                else:
                    t_in = n_to + "_" + carrier

                dotG.add_edge(
                    pydot.Edge(
                        src=t_out,
                        dst=t_in,
                        color='"{0}:invis:{0}"'.format(col["e"][carrier]),
                        fontcolor=col["e"][carrier],
                        label=edgelabel,
                        taillabel=taillabel,
                        headlabel=headlabel,
                    )
                )

    if filename is not None:
        # prog='dot' gives the best layout.
        dotG.write(filename, prog=prog, format="png")
    return dotG
