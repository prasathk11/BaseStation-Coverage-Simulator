import tkinter as tk
from simulator import BaseStationSimulator


def main():
    """Initialize and run the simulator."""
    root = tk.Tk()
    simulator = BaseStationSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
