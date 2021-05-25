import pyomo.environ as pyo
import logging

class Device:
    "Parent class from which all device types derive"

    # Define common fields and methods
    dev_id = None
    params = {}
    carrier_in = []
    carrier_out = []
    serial = []
    dev_constraints = None
    pyomo_model = None
    optimiser = None # Oogeso optimiser object

    def __init__ (self,pyomo_model,dev_id,dev_data,optimiser):
        """Device object constructor"""
        self.dev_id = dev_id
        self.pyomo_model = pyomo_model
        self.params = dev_data
        self.optimiser = optimiser

    def setInitValues(self):
        dev_id = self.dev_id
        pyomo_model = self.pyomo_model
        dev_data = self.params
        if 'E_init' in dev_data:
            pyomo_model.paramDeviceEnergyInitially[dev_id] = dev_data['E_init']
        if 'isOn_init' in dev_data:
            pyomo_model.paramDeviceIsOnInitially[dev_id] = dev_data['isOn_init']
        if 'P_init' in dev_data:
            pyomo_model.paramDevicePowerInitially[dev_id] = dev_data['P_init']

    def _rule_devicePmax(self,model,t):
        # max/min power (zero if device is not on)
        maxValue = self.params['Pmax']
        if 'profile' in self.params:
            # use an availability profile if provided
            extprofile = self.params['profile']
            maxValue = maxValue*model.paramProfiles[extprofile,t]
        power = self.getPowerVar(t)
        expr = ( power <= maxValue)
        return expr

    def _rule_devicePmin(self,model,t):
        params_dev=self.params
        minValue = params_dev['Pmin']
        if 'profile' in params_dev:
            # use an availability profile if provided
            extprofile = params_dev['profile']
            minValue = minValue*model.paramProfiles[extprofile,t]
        power = self.getPowerVar(t)
        expr = (power >= minValue)
        return expr

    def _rule_deviceQmax(self,model,t):
        params_dev=self.params
        maxValue = params_dev['Qmax']
        if 'profile' in params_dev:
            # use an availability profile if provided
            extprofile = params_dev['profile']
            maxValue = maxValue*model.paramProfiles[extprofile,t]
        flow = self.getFlowVar(t)
        expr = ( flow <= maxValue)
        return expr


    def _rule_deviceQmin(self,model,t):
        params_dev=self.params
        minValue = params_dev['Qmin']
        if 'profile' in params_dev:
            # use an availability profile if provided
            extprofile = params_dev['profile']
            minValue = minValue*model.paramProfiles[extprofile,t]
        flow = self.getFlowVar(t)
        expr = (flow >= minValue)
        return expr

    def _rule_ramprate(self,model,t):
        '''power ramp rate limit'''
        dev = self.dev_id
        param_dev = self.params
        param_generic = self.optimiser.optimisation_parameters

        # If no ramp limits have been specified, skip constraint
        if ('maxRampUp' not in param_dev):
            return pyo.Constraint.Skip
        if (t>0):
            p_prev = self.getPowerVar(t-1)
        else:
            p_prev = model.paramDevicePowerInitially[dev]
        p_this = self.getPowerVar(t)
        deltaP = (self.getPowerVar(t) - p_prev)
        delta_t = param_generic['time_delta_minutes']
        maxP = param_dev['Pmax']
        max_neg = -param_dev['maxRampDown']*maxP*delta_t
        max_pos = param_dev['maxRampUp']*maxP*delta_t
        expr = pyo.inequality(max_neg, deltaP, max_pos)
        return expr


    def defineConstraints(self):
        """Returns a set of constraints for the device."""

        # Generic constraints common for all device types:

        if 'Pmax' in self.params:
            constrDevicePmax = pyo.Constraint(self.pyomo_model.setHorizon,
                rule=self._rule_devicePmax)
            setattr(self.pyomo_model,'constr_{}_{}'.format(
                self.dev_id,'Pmax'),constrDevicePmax)
        if 'Pmin' in self.params:
            constrDevicePmin = pyo.Constraint(
                self.pyomo_model.setHorizon,rule=self._rule_devicePmin)
            setattr(self.pyomo_model,'constr_{}_{}'.format(
                self.dev_id,'Pmin'),constrDevicePmin)
        if 'Qmax' in self.params:
            constrDeviceQmax = pyo.Constraint(
            self.pyomo_model.setHorizon,rule=self._rule_deviceQmax)
            setattr(self.pyomo_model,'constr_{}_{}'.format(
                self.dev_id,'Qmax'),constrDeviceQmax)
        if 'Qmin' in self.params:
            constrDeviceQmin = pyo.Constraint(
            self.pyomo_model.setHorizon,rule=self._rule_deviceQmin)
            setattr(self.pyomo_model,'constr_{}_{}'.format(
                self.dev_id,'Qmin'),constrDeviceQmin)

        if 'maxRampUp' in self.params:
            constrDevice_ramprate = pyo.Constraint(
                self.pyomo_model.setHorizon, rule=self._rule_ramprate)
            setattr(self.pyomo_model,'constr_{}_{}'.format(
                self.dev_id,'ramprate'),constrDevice_ramprate)



    def getPowerVar(self,t):
        #logging.debug("Device: no getPowerVar defined for {}"
        #    .format(self.dev_id))
        #raise NotImplementedError()
        return 0

    def getFlowVar(self,t):
        raise NotImplementedError()

    def getMaxP(self,t):
        model=self.pyomo_model
        maxValue = self.params['Pmax']
        if 'profile' in self.params:
            extprofile = self.params['profile']
            maxValue = maxValue*model.paramProfiles[extprofile,t]
        return maxValue

    def compute_CO2(self,timesteps):
        return 0

    def compute_export(self,value,carriers,timesteps):
        '''Compute average export (volume or revenue)

        Parameters:
        -----------
        value : str
            "revenue" (€/s) or "volume" (Sm3oe/s)
        carriers : list of carriers ("gas","oil","el")
        timesteps : list of timesteps
        '''
        carriers_in = self.carrier_in
        carriers_incl = [v for v in carriers if v in carriers_in]
        sumValue = 0
        for c in carriers_incl:
            # flow in m3/s, price in $/m3
            price_param = 'price.{}'.format(c)
            if price_param in self.params:
                inflow = sum(
                    self.pyomo_model.varDeviceFlow[self.dev_id,c,'in',t]
                    for t in timesteps)
                if value=="revenue":
                    sumValue += inflow*self.params[price_param]
                elif value=="volume":
                    volumefactor=1
                    if c=='gas':
                        volumefactor = 1/1000 # Sm3 to Sm3oe
                    sumValue += inflow*volumefactor
        return sumValue
    #computeStartupCosts(...)
    #computeOperatingCosts(...)

    def compute_elReserve(self,t):
        '''Compute available reserve power from this device

        device parameter "reserve_factor" specifies how large part of the
        available capacity should count towards the reserve (1=all, 0=none)
        '''
        model = self.pyomo_model
        rf = 1
        loadreduction = 0
        cap_avail = 0
        p_generating = 0
        if 'el' in self.carrier_out:
            # Generators and storage
            maxValue = self.getMaxP(t)
            if ('reserve_factor' in self.params):
                # safety margin - only count a part of the forecast power
                # towards the reserve, relevant for wind power
                # (equivalently, this may be seen as increaseing the
                # reserve margin requirement)
                reserve_factor=self.params['reserve_factor']
                maxValue = maxValue*reserve_factor
                if reserve_factor==0:
                    # no reserve contribution
                    rf = 0
            cap_avail = rf*maxValue
            p_generating = rf*model.varDeviceFlow[self.dev_id,'el','out',t]
        elif 'el' in self.carrier_in:
            # Loads (only consider if resere factor has been set)
            if ('reserve_factor' in self.params):
                # load reduction possible
                f_lr = self.params['reserve_factor']
                loadreduction = f_lr*model.varDeviceFlow[self.dev_id,'el','in',t]
        reserve = {'capacity_available': cap_avail,
                    'capacity_used': p_generating,
                    'loadreduction_available': loadreduction}
        return reserve

    # only gas turbine has non-zero start/stop costs
    def compute_startup_costs(self,timesteps):
        return 0

    def compute_operatingCosts(self,timesteps):
        '''average operating cost within selected timespan'''
        sumCost = 0
        if 'opCost' in self.params:
            opcost = self.params['opCost']
            for t in self.pyomo_model.setHorizon:
                varP = self.getPowerVar(t)
                sumCost += opcost*varP
        # average per sec (simulation timestep drops out)
        avgCost = sumCost/len(timesteps)
        return avgCost

    def compute_costForDepletedStorage(self,timesteps):
        return 0

    def getProfile(self):
        '''Get device profile as list of values, or None if no profile is used'''
        profile = None
        if 'profile' in self.params:
            prof_id = self.params['profile']
            profile = self.pyomo_model.paramProfiles[prof_id,:].value
        return profile
