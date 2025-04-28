# water_control.py
import time

class WaterControl:
    def __init__(self):
        self.is_dispensing = False

    def start_dispensing(self):
        self.is_dispensing = True
        print("Water dispensing started")
        # Here you would add code to control the actual water flow

    def stop_dispensing(self):
        self.is_dispensing = False
        print("Water dispensing stopped")
        # Here you would add code to stop the actual water flow

    def get_flow_rate(self):
        # This method would interact with a flow sensor to get the actual flow rate
        # For now, we'll return a dummy value
        return 0.5  # L/min