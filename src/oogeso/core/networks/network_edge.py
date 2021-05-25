import pyomo.environ as pyo
import logging
import numpy as np
import scipy
from . import electricalsystem as el_calc

class NetworkEdge:
    "Network edge"
    node_id = None
    pyomo_model = None
    params = {}
    optimiser = None

    def __init__ (self,pyomo_model,edge_id,edge_data,optimiser):
        self.edge_id = edge_id
        self.pyomo_model = pyomo_model
        self.params = edge_data
        self.optimiser = optimiser



    logging.info("TODO: equation for flow vs pressure of liquids")
    def _rule_edgeFlowEquations(self,model,t):
        '''Flow as a function of node values (voltage/pressure)'''
        edge = self.edge_id
        params_edge = self.params
        carrier = params_edge['type']
        n_from = params_edge['nodeFrom']
        n_to = params_edge['nodeTo']

        if carrier == 'el':
            '''power flow vs voltage angle difference
            Linearised power flow equations (DC power flow)'''
            baseMVA = el_calc.elbase['baseMVA']
            baseAngle = el_calc.elbase['baseAngle']
            lhs = model.varEdgeFlow[edge,t]
            lhs = lhs/baseMVA
            rhs = 0
            #TODO speed up creatioin of constraints - remove for loop
            n2s = [k[1]  for k in self.optimiser.elFlowCoeffDA.keys() if k[0]==edge]
            for n2 in n2s:
                rhs += self.optimiser.elFlowCoeffDA[edge,n2]*(
                        model.varElVoltageAngle[n2,t]*baseAngle)
            return (lhs==rhs)

        elif carrier in ['gas','wellstream','oil','water']:
            p1 = model.varPressure[(n_from,carrier,'out',t)]
            p2 = model.varPressure[(n_to,carrier,'in',t)]
            Q = model.varEdgeFlow[edge,t]
            if 'num_pipes' in params_edge:
                num_pipes = params_edge['num_pipes']
                logging.debug("{},{}: {} parallel pipes"
                    .format(edge,t,num_pipes))
                Q = Q/num_pipes
            p2_computed = self.compute_edge_pressuredrop(
                model,p1=p1,Q=Q,linear=True)
            return (p2==p2_computed)
        else:
            #Other types of edges - no constraints other than max/min flow
            return pyo.Constraint.Skip


    def _rule_edgePmaxmin(self,model,t):
        params_edge = self.params
        edge = self.edge_id
        edgetype = params_edge['type']
        expr = pyo.Constraint.Skip #default if Pmax/Qmax has not been set
        if edgetype in ['el','heat']:
            if 'Pmax' in params_edge:
                expr = pyo.inequality(-params_edge['Pmax'],
                        model.varEdgeFlow[edge,t],
                        params_edge['Pmax'])
        elif edgetype in ['wellstream','gas','oil','water','hydrogen']:
            if 'Qmax' in params_edge:
                expr = (model.varEdgeFlow[edge,t] <=
                        params_edge['Qmax'])
        else:
            raise Exception("Unknown edge type ({})".format(edgetype))
        return expr


    def defineConstraints(self):
        """Returns the set of constraints for the node."""

        if (('Pmax' in self.params) or ('Qmax' in self.params)):
            constrEdgeBounds = pyo.Constraint(
                self.pyomo_model.setHorizon,rule=self._rule_edgePmaxmin)
            setattr(self.pyomo_model,'constrE_{}_{}'.format(
                self.edge_id,'bounds'),constrEdgeBounds)

        constr_flow = pyo.Constraint(self.pyomo_model.setHorizon,
            rule=self._rule_edgeFlowEquations)
        setattr(self.pyomo_model,'constrE_{}_{}'.format(
            self.edge_id,'flow'),constr_flow)


    def _compute_exps_and_k(self,param_carrier):
        '''Derive exp_s and k parameters for Weymouth equation'''
        param_edge = self.params
        # gas pipeline parameters - derive k and exp(s) parameters:
        ga=param_carrier

        #if 'temperature_K' in model.paramEdge[edge]:
        temp = param_edge['temperature_K']
        height_difference = param_edge['height_m']
        length = param_edge['length_km']
        diameter = param_edge['diameter_mm']
        s = 0.0684 * (ga['G_gravity']*height_difference
                        /(temp*ga['Z_compressibility']))
        if s>0:
            # height difference - use equivalent length
            sfactor= (np.exp(s)-1)/s
            length = length*sfactor

        k = (4.3328e-8*ga['Tb_basetemp_K']/ga['Pb_basepressure_MPa']
            *(ga['G_gravity']*temp*length*ga['Z_compressibility'])**(-1/2)
            *diameter**(8/3))
        exp_s = np.exp(s)
        return exp_s,k


    def compute_edge_pressuredrop(self,model,p1,
            Q,method=None,linear=False):
        '''Compute pressure drop in pipe

        parameters
        ----------
        model : pyomo optimisation model
        p1 : float
            pipe inlet pressure (MPa)
        Q : float
            flow rate (Sm3/s)
        method : string
            None, weymouth, darcy-weissbach
        linear : boolean
            whether to use linear model or not

        Returns
        -------
        p2 : float
            pipe outlet pressure (MPa)'''

        edge = self.edge_id
        param_edge = self.params
        height_difference = param_edge['height_m']
        method = None
        carrier = param_edge['type']
        param_carrier = self.optimiser.all_carriers[carrier].params
        if 'pressure_method' in param_carrier:
            method = param_carrier['pressure_method']

        n_from = param_edge['nodeFrom']
        n_to = param_edge['nodeTo']
        n_from_obj = self.optimiser.all_nodes[n_from]
        n_to_obj = self.optimiser.all_nodes[n_to]
        if (('pressure.{}.out'.format(carrier) in n_from_obj.params) &
            ('pressure.{}.in'.format(carrier) in n_to_obj.params)):
            p0_from = n_from_obj.params['pressure.{}.out'.format(carrier)]
            p0_to = n_to_obj.params['pressure.{}.in'.format(carrier)]
            if (linear & (p0_from==p0_to)):
                method=None
                logging.debug(("{}-{}: Pipe without pressure drop"
                    " ({} / {} MPa)").format(n_from,n_to,p0_from,p0_to))
        elif linear:
            # linear equations, but nominal values not given - assume no drop
            method=None

        if method is None:
            # no pressure drop
            p2 = p1
            return p2

        elif method=='weymouth':
            '''
            Q = k * sqrt( Pin^2 - e^s Pout^2 ) [Weymouth equation, nonlinear]
                => Pout = sqrt(Pin^2-Q^2/k^2)/e^2
            Q = c * (Pin_0 Pin - e^s Pout0 Pout) [linearised version]
                => Pout = (Pin0 Pin - Q/c)/(e^s Pout0)
            c = k/sqrt(Pin0^2 - e^s Pout0^2)

            REFERENCES:
            1) E Sashi Menon, Gas Pipeline Hydraulics, Taylor & Francis (2005),
            https://doi.org/10.1201/9781420038224
            2) A Tomasgard et al., Optimization  models  for  the  natural  gas
            value  chain, in: Geometric Modelling, Numerical Simulation and
            Optimization. Springer Verlag, New York (2007),
            https://doi.org/10.1007/978-3-540-68783-2_16
            '''
            exp_s,k = self._compute_exps_and_k(param_carrier)
            logging.debug("pipe {}: exp_s={}, k={}"
                .format(edge,exp_s,k))
            if linear:
                p_from=p1
    #                p_from = model.varPressure[(n_from,carrier,'out',t)]
    #                p_to = model.varPressure[(n_to,carrier,'in',t)]
                X0 = p0_from**2-exp_s*p0_to**2
    #                logging.info("edge {}-{}: X0={}, p1={},Q={}"
    #                    .format(n_from,n_to,X0,p1,Q))
                coeff = k*(X0)**(-1/2)
    #                Q_computed = coeff*(p0_from*p_from - exp_s*p0_to*p_to)
                p2 = (p0_from*p_from-Q/coeff)/(exp_s*p0_to)
            else:
                # weymouth eqn (non-linear)
                p2 = 1/exp_s*(p1**2-Q**2/k**2)**(1/2)

        elif method=='darcy-weissbach':
            grav=9.98 #m/s^2
            rho = param_carrier['rho_density']
            D = param_edge['diameter_mm']/1000
            L = param_edge['length_km']*1000

            if (('viscosity' in param_carrier)
                & (not linear)):
                #compute darcy friction factor from flow rate and viscosity
                mu = param_carrier['viscosity']
                Re = 2*rho*Q/(np.pi*mu*D)
                f = 1/(0.838*scipy.special.lambertw(0.629*Re))**2
                f = f.real
            elif 'darcy_friction' in param_carrier:
                f = param_carrier['darcy_friction']
                Re = None
            else:
                raise Exception(
                    "Must provide viscosity or darcy_friction for {}"
                    .format(carrier))
            if linear:
                p_from = p1
                k = np.sqrt(np.pi**2 * D**5/(8*f*rho*L))
                sqrtX = np.sqrt(p0_from*1e6 - p0_to*1e6
                                - rho*grav*height_difference)
                Q0 = k*sqrtX
                logging.debug(("derived pipe ({}) flow rate:"
                    " Q={}, linearQ0={:5.3g},"
                    " friction={:5.3g}")
                         .format(edge,Q,Q0,f))
                p2 = p_from - 1e-6*(Q-Q0)*2*sqrtX/k - (p0_from-p0_to)
                # linearised darcy-weissbach:
                # Q = Q0 + k/(2*sqrtX)*(p_from-p_to - (p0_from-p0_to))
            else:
                # darcy-weissbach eqn (non-linear)
                p2 = 1e-6*(
                    p1*1e6
                    - rho*grav*height_difference
                    - 8*f*rho*L*Q**2/(np.pi**2*D**5) )
        else:
            raise Exception("Unknown pressure drop calculation method ({})"
                            .format(method))
        return p2

def darcy_weissbach_Q(p1,p2,f,rho,diameter_mm,
        length_km,height_difference_m=0):
    '''compute flow rate from darcy-weissbach eqn

    parameters
    ----------
    p1 : float
        pressure at pipe input (Pa)
    p2 : float
        pressure at pipe output (Pa)
    f : float
        friction factor
    rho : float
        fluid density (kg/m3)
    diameter_mm : float
        pipe inner diameter (mm)
    length_km : float
        pipe length (km)
    height_difference_m : float
        height difference output vs input (m)

    '''

    grav = 9.98
    L=length_km*1000
    D=diameter_mm/1000
    k = 8*f*rho*L/(np.pi**2*D**5)
    Q = np.sqrt( ((p1-p2)*1e6 - rho*grav*height_difference_m) / k )
    return Q
