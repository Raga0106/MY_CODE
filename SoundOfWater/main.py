# main.py
import numpy as np
import logging
from data_collector import DataCollector
from ml_model import WaterLevelPredictor
from physics_engine import AcousticSimulator
from user_interface import UserInterface
from water_control import WaterControl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmartWaterDispenserSystem:
    def __init__(self):
        self.acoustic_simulator = AcousticSimulator()
        self.ml_model = WaterLevelPredictor()
        self.data_collector = DataCollector()
        self.water_control = WaterControl()
        self.user_interface = UserInterface()
        self.samples_since_last_training = 0
        self.training_frequency = 100  # Train every 100 samples
        self.performance_threshold = 0.8  # R-squared threshold
        logging.info("Smart Water Dispenser System initialized")

    def run(self):
        try:
            self.initial_training()

            while True:
                target_percentage = self.user_interface.get_target_percentage()
                logging.info(f"Target percentage set to {target_percentage}%")
                self.water_control.start_dispensing()

                current_height = 0
                while current_height < self.acoustic_simulator.container_height:
                    try:
                        current_height += self.water_control.get_flow_rate() * 0.1  # Assuming 0.1 seconds between measurements
                        acoustic_data = self.acoustic_simulator.simulate(current_height)
                        predicted_level = self.ml_model.predict(acoustic_data)

                        if predicted_level >= target_percentage:
                            self.water_control.stop_dispensing()
                            break

                        self.data_collector.record(acoustic_data, predicted_level)
                        self.samples_since_last_training += 1

                        if self.samples_since_last_training >= self.training_frequency:
                            self.periodic_training()
                    except Exception as e:
                        logging.error(f"Error during dispensing: {str(e)}")
                        self.water_control.stop_dispensing()
                        break

                self.user_interface.display_result(predicted_level)
        except KeyboardInterrupt:
            logging.info("System shutdown initiated by user")
        finally:
            self.cleanup()

    def initial_training(self):
        logging.info("Generating initial training data...")
        initial_data = self.generate_simulated_data(1000)  # Generate 1000 simulated data points
        self.data_collector.record_bulk(initial_data)
        self.train_model()

    def generate_simulated_data(self, num_samples):
        simulated_data = []
        for _ in range(num_samples):
            current_height = np.random.uniform(0, self.acoustic_simulator.container_height)
            acoustic_data = self.acoustic_simulator.simulate(current_height)
            level_percentage = (current_height / self.acoustic_simulator.container_height) * 100
            simulated_data.append({**acoustic_data, 'level': level_percentage})
        return simulated_data

    def periodic_training(self):
        logging.info("Performing periodic training...")
        training_success = self.train_model()
        if training_success:
            self.evaluate_model_performance()
        self.samples_since_last_training = 0

    def train_model(self):
        training_data = self.data_collector.get_data()
        return self.ml_model.train_model(training_data)

    def evaluate_model_performance(self):
        test_data = self.generate_simulated_data(200)  # Generate 200 test samples
        X_test = np.array([[d['frequency'], d['amplitude']] for d in test_data])
        y_test = np.array([d['level'] for d in test_data])

        mse, r2 = self.ml_model.evaluate(X_test, y_test)
        if r2 < self.performance_threshold:
            logging.warning(f"Model performance below threshold. R-squared: {r2}")
            self.adjust_model_or_data_collection()
        else:
            logging.info(f"Model performance acceptable. R-squared: {r2}")

    def adjust_model_or_data_collection(self):
        logging.info("Adjusting data collection strategy...")
        self.training_frequency = max(50, self.training_frequency - 10)  # Train more frequently, but not less than every 50 samples
        logging.info(f"New training frequency: every {self.training_frequency} samples")

    def cleanup(self):
        self.water_control.stop_dispensing()
        self.data_collector.save_data()
        self.ml_model.save_model()
        logging.info("Cleanup completed, system shut down")

if __name__ == "__main__":
    system = SmartWaterDispenserSystem()
    system.run()