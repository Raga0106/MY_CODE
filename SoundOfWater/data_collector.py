# data_collector.py
import json
from datetime import datetime
import os
import numpy as np

class DataCollector:
    def __init__(self):
        self.data = []
        self.filename = self._generate_filename()

    def _generate_filename(self):
        base_name = "training_data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.json"

    def _convert_to_serializable(self, data):
        if isinstance(data, dict):
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(v) for v in data]
        elif isinstance(data, np.float32):
            return float(data)
        else:
            return data

    def record(self, acoustic_data, predicted_level):
        serializable_data = self._convert_to_serializable({**acoustic_data, 'level': predicted_level})
        self.data.append(serializable_data)
        self.save_data()

    def get_data(self):
        return self.data

    def save_data(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f)
        except OSError as e:
            print(f"Error saving data: {e}")
            simple_filename = "training_data.json"
            try:
                with open(simple_filename, 'w') as f:
                    json.dump(self.data, f)
                print(f"Data saved to {simple_filename}")
            except OSError as e:
                print(f"Failed to save data: {e}")

    def load_data(self, filename):
        try:
            with open(filename, 'r') as f:
                self.data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading data: {e}")
            self.data = []

    def record_bulk(self, data):
        serializable_data = [self._convert_to_serializable(d) for d in data]
        self.data.extend(serializable_data)
        self.save_data()
