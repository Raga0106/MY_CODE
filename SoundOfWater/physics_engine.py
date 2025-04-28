# physics_engine.py
import numpy as np

class AcousticSimulator:
    def __init__(self, container_height=20, base_frequency=1000):
        self.container_height = container_height  # cm
        self.base_frequency = base_frequency  # Hz
        self.sound_speed = 343  # m/s (speed of sound in air at room temperature)
        self.material_properties = {
            "glass": {"density": 2500, "youngs_modulus": 70e9},
            "plastic": {"density": 1000, "youngs_modulus": 2e9},
            "metal": {"density": 7800, "youngs_modulus": 200e9}
        }
        self.current_material = "glass"

    def simulate(self, current_height):
        if current_height < 0 or current_height > self.container_height:
            raise ValueError("Invalid current height")

        time = 2 * (self.container_height - current_height) / self.sound_speed
        frequency = self.base_frequency + (1 / time)

        frequency += np.random.normal(0, frequency * 0.01)  # 1% standard deviation

        material_props = self.material_properties[self.current_material]
        impedance = np.sqrt(material_props["density"] * material_props["youngs_modulus"])
        amplitude = 1 / impedance

        amplitude += np.random.normal(0, amplitude * 0.05)  # 5% standard deviation

        return {'frequency': frequency, 'amplitude': amplitude}

    def set_container_parameters(self, height, material):
        if material not in self.material_properties:
            raise ValueError(f"Unsupported material: {material}")

        self.container_height = height
        self.current_material = material

    def add_material(self, material_name, density, youngs_modulus):
        self.material_properties[material_name] = {
            "density": density,
            "youngs_modulus": youngs_modulus
        }
