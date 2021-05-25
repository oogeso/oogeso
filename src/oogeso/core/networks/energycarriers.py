
class EnergyCarrier:
    "Energy carrier"
    carrier_id = None
    pyomo_model = None
    params = {}

    def __init__ (self,pyomo_model,carrier_id,carrier_data):
        self.carrier_id = carrier_id
        self.pyomo_model = pyomo_model
        self.params = carrier_data
