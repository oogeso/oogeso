import pyomo.environ as pyo
import typing

# if typing.TYPE_CHECKING:
from ...dto.oogeso_input_data_objects import EdgeData


class Edge:
    "Network edge"

    def __init__(self, edge_data_object: EdgeData):
        self.id = edge_data_object.id
        self.edge_data = edge_data_object  # Edge data object as defined in the DTO
        self.edges = {}

    def defineConstraints(self, pyomo_model, piecewise_repn):
        """Builds constraints for the edge"""

        constr_edge_bounds = pyo.Constraint(
            pyomo_model.setHorizon, rule=self._ruleEdgeFlowMaxMin
        )
        setattr(
            pyomo_model,
            "constrE_{}_{}".format(self.id, "bounds"),
            constr_edge_bounds,
        )

        # Losses (flow out of edge less than into edge)
        if self.has_loss():

            # First, connecting variables (flow in different directions and loss variables)
            constr_loss = pyo.Constraint(
                pyomo_model.setHorizon,
                pyo.RangeSet(1, 2),
                rule=self._ruleEdgeFlowAndLoss,
            )
            setattr(pyomo_model, "constrE_{}_{}".format(self.id, "loss"), constr_loss)
            # Then, add equations for losses vs power flow (piecewise linear equations):
            for i in [1, 2]:
                constr_loss_function = self._loss_function_constraint(
                    i, pyomo_model, piecewise_repn
                )
                setattr(
                    pyomo_model,
                    "constrE_{}_{}_{}".format(self.id, "lossfunction", i),
                    constr_loss_function,
                )

    def _ruleEdgeFlowMaxMin(self, model, t):
        edge = self.id
        # if non-bidirectional, lower bound=0, if bidirectional, bound=+- pmax
        if self.edge_data.flow_max is None:
            if self.edge_data.bidirectional:
                # unconstrained
                expr = pyo.Constraint.Skip
            else:
                # one-directional, so lower bound 0 (no upper bound specified)
                # expr = pyo.inequality(0, model.varEdgeFlow[edge, t], None)
                expr = model.varEdgeFlow[edge, t] >= 0
        else:
            pmax = self.edge_data.flow_max
            if self.edge_data.bidirectional:
                expr = pyo.inequality(-pmax, model.varEdgeFlow[edge, t], pmax)
            else:
                expr = pyo.inequality(0, model.varEdgeFlow[edge, t], pmax)
        return expr

    def has_loss(self):
        hasloss = hasattr(self.edge_data, "power_loss_function") and (
            self.edge_data.power_loss_function is not None
        )
        return hasloss

    def _ruleEdgeFlowAndLoss(self, model, t, i):
        """Split edge flow into positive and negative part, for loss calculations"""
        edge = self.id
        if i == 1:
            expr = (
                model.varEdgeFlow[edge, t]
                == model.varEdgeFlow12[edge, t] - model.varEdgeFlow21[edge, t]
            )
        elif i == 2:
            expr = (
                model.varEdgeLoss[edge, t]
                == model.varEdgeLoss12[edge, t] + model.varEdgeLoss21[edge, t]
            )
        return expr

    def _loss_function_constraint(self, i, pyomo_model, piecewise_repn):

        # Piecewise constraints require independent variable to be bounded:
        pyomo_model.varEdgeFlow12[self.id, :].setub(self.edge_data.flow_max)
        pyomo_model.varEdgeFlow21[self.id, :].setub(self.edge_data.flow_max)
        # Losses on cables are: P_loss = R/V^2 * P^2, i.e. quadratic function of power flow
        # Losses in transformers are: P_loss = ...
        lookup_table = self.edge_data.power_loss_function
        pw_x = lookup_table[0]
        pw_y = lookup_table[1]
        # # If we use loss function giving loss fraction instead of absolute loss in MW:
        # pw_y_fraction = lookup_table[1]
        # # Lookup table gives losses as a fraction, so to get absolute values we need
        # # to multiply by power transfer
        # # NOTE: This changes a linear curve to a second order curve, so need more than
        # # two x-points to represent it properly.
        # pw_y = [pw_y_fraction[i] * pw_x[i] for i in len(pw_x)]
        if i == 1:
            var_x = pyomo_model.varEdgeFlow12
            var_y = pyomo_model.varEdgeLoss12
        elif i == 2:
            var_x = pyomo_model.varEdgeFlow21
            var_y = pyomo_model.varEdgeLoss21
        constr_penalty = pyo.Piecewise(
            [self.id],
            pyomo_model.setHorizon,
            var_y,
            var_x,
            pw_repn=piecewise_repn,
            pw_constr_type="EQ",
            pw_pts=pw_x,
            f_rule=pw_y,
        )
        return constr_penalty
