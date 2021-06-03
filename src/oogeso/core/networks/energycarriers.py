from ...dto.oogeso_input_data_objects import CarrierData


class EnergyCarrier:
    "Energy carrier"

    def __init__(self, carrier_data: CarrierData):
        self.carrier_id = carrier_data.id
        self.carrier_data = carrier_data
