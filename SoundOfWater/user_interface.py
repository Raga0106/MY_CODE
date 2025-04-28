# user_interface.py
class UserInterface:
    def get_target_percentage(self):
        while True:
            try:
                percentage = float(input("Enter target water level percentage (0-100): "))
                if 0 <= percentage <= 100:
                    return percentage
                else:
                    print("Please enter a value between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def display_result(self, final_level):
        print(f"Final water level: {final_level:.2f}%")

    def display_error(self, message):
        print(f"Error: {message}")

    def continue_dispensing(self):
        while True:
            response = input("Do you want to dispense more water? (y/n): ").lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
