import pydot
import dot2tex as d2t

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
    node_fs = 30,
    dev_fs = 16,
    edge_fs = 16,
    carrier_fs = 16,
    fn = "helvetica",
    names = None,
    exclude = None,
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
    names : dict
        A dictionary with new names for the network components
    exclude : list
        Names of devices to exclude from adding to the graph
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
            "diesel": "black",
            "heat": "darkgreen",
            "wellstream": "brown",
            "oil": "black",
            "water": "blue4",
            "hydrogen": "deepskyblue2",
        },
        "e": {
            "el": "red",
            "gas": "orange",
            "diesel": "black",
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
        if exclude is not None and n_id in exclude:
            pass
        else:
            if names is not None and n_id in names:
                cluster_name = names[n_id]
            else:
                cluster_name = n_id  
            cluster = pydot.Cluster(graph_name=n_id, label=cluster_name, style="filled", color=col["cluster"], fontsize=node_fs, fontname = fn)
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
                        if exclude is not None and d in exclude:
                            pass
                        else:
                            dev_model = dev_obj.dev_data.model
                            if names is not None and d in names:
                                d_id = names[d]
                            else:
                                d_id = d
                            if names is not None and dev_model in names:
                                dev_model = names[dev_model]
                            devlabel = "<{}<BR/>{}>".format(d_id, dev_model)
                            if plot_device_name:
                                dev_name = dev_obj.dev_data.name
                                if names is not None and dev_name in names:
                                    dev_name = names[dev_name]
                                devlabel = "<{} {}>".format(devlabel, dev_name)
                            carriers_in = dev_obj.carrier_in
                            carriers_out = dev_obj.carrier_out
                            carriers_in_lim = list(set(carriers_in) & set(carriers))
                            carriers_out_lim = list(set(carriers_out) & set(carriers))
                            if (carriers_in_lim != []) or (carriers_out_lim != []):
                                gr_devices.add_node(pydot.Node(d, shape="box", color=col["d"], style="rounded, filled", margin="0.2, 0.15", label=devlabel, fontsize=dev_fs, fontname=fn))
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
                                if names is not None and devedgelabel in names:
                                    devedgelabel = names[devedgelabel]
                                dotG.add_edge(
                                    pydot.Edge(
                                        dst=d,
                                        src=n_in,
                                        color=col["e"][carrier],
                                        fontcolor=col["e"][carrier],
                                        label=devedgelabel,
                                        fontsize=edge_fs,
                                        fontname = fn + " bold",
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
                                if names is not None and devedgelabel in names:
                                    devedgelabel = names[devedgelabel]
                                dotG.add_edge(
                                    pydot.Edge(
                                        dst=n_out,
                                        src=d,
                                        color=col["e"][carrier],
                                        fontcolor=col["e"][carrier],
                                        label=devedgelabel,
                                        fontsize=edge_fs,
                                        fontname = fn + " bold",
                                    )
                                )

                    # add in/out terminals
                    label_in = carrier  # + "_in "
                    label_out = carrier  # + supp + " "
                    if names is not None and label_in in names:
                        label_in = names[label_in]
                    if names is not None and label_out in names:
                        label_out = names[label_out]
                    if timestep is None:
                        pass
                    elif carrier in ["gas", "wellstream", "oil", "water"]:
                        label_in += number_format.format(res.terminal_pressure[(n_id, carrier, "in", timestep)])
                        label_out += number_format.format(res.terminal_pressure[(n_id, carrier, "out", timestep)])
                    elif carrier == "el":
                        if optimiser.all_networks["el"].carrier_data.powerflow_method == "dc-pf":
                            label_in += number_format.format(res.el_voltage_angle[(n_id, timestep)])
                            label_out += number_format.format(res.el_voltage_angle[(n_id, timestep)])
                    # Add two terminals if there are serial devices, otherwise one:
                    if carrier in node_obj.devices_serial:
                        terms_in.add_node(
                            pydot.Node(
                                name=n_id + "_" + carrier + "_in",
                                color=col["t"][carrier],
                                label=label_in,
                                shape="box",
                                fontsize=carrier_fs,
                                fontname = fn,
                            )
                        )
                        terms_out.add_node(
                            pydot.Node(
                                name=n_id + "_" + carrier + "_out",
                                color=col["t"][carrier],
                                label=label_out,
                                shape="box",
                                fontsize=carrier_fs,
                                fontname = fn,
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
                                    fontsize=carrier_fs,
                                    fontname = fn,
                                )
                            )
                        else:
                            terms_in.add_node(
                                pydot.Node(
                                    name=n_id + "_" + carrier,
                                    color=col["t"][carrier],
                                    label=label_out,
                                    shape="box",
                                    fontsize=carrier_fs,
                                    fontname = fn,
                                )
                            )

            cluster.add_subgraph(terms_in)
            cluster.add_subgraph(gr_devices)
            cluster.add_subgraph(terms_out)
            dotG.add_subgraph(cluster)

    # plot all edges (per carrier):
    for carrier in carriers:
        for i, edge_obj in optimiser.all_edges.items():
            if exclude is not None and i in exclude:
                pass
            else:
                edge_data = edge_obj.edge_data
                if edge_data.carrier == carrier:
                    headlabel = ""
                    taillabel = ""
                    if names is not None and headlabel in names:
                        headlabel = names[headlabel]
                    if names is not None and taillabel in names:
                        taillabel = names[taillabel]
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
                            fontsize=edge_fs,
                            fontname = fn + " bold",
                        )
                    )

    if filename is not None:
        # prog='dot' gives the best layout.
        dotG.write(filename, prog=prog, format="png")
    return dotG
