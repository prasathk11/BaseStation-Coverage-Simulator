import tkinter as tk
from simulator import BaseStationSimulator


def main():
    root = tk.Tk()
    root.geometry("1400x850")
    simulator = BaseStationSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()