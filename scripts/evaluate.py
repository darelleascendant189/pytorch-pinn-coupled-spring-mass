"""
Evaluation script for the PINN model.

Compares the PINN predictions against a traditional ODE solver (SciPy's solve_ivp).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
import os
import logging

# Ensure src directory is in path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import PINN
from src.utils import setup_logging, get_device

def get_ode_system_func(N: int, alpha: float):
    """
    Creates the system function for scipy.integrate.solve_ivp.
    
    State vector y = [x_1, ..., x_N, v_1, ..., v_N]
    
    Args:
        N (int): Number of masses.
        alpha (float): System constant k/m.
        
    Returns:
        function: A function f(t, y) that returns dy/dt.
    """
    def mass_spring_system(t, y):
        dydt = np.zeros(N * 2)
        x = y[:N]  # Positions
        v = y[N:]  # Velocities

        for i in range(N):
            # Calculate acceleration 'a' for mass i
            x_left = x[i - 1] if i > 0 else 0.0
            x_right = x[i + 1] if i < N - 1 else 0.0
            
            a = alpha * (x_left - 2 * x[i] + x_right)
            
            dydt[i] = v[i]      # dx_i/dt = v_i
            dydt[N + i] = a   # dv_i/dt = a_i
            
        return dydt
    
    return mass_spring_system

def main(args):
    """
    Main evaluation function.
    """
    setup_logging(args.log_file)
    device = get_device()
    
    logging.info(f"Starting evaluation for {args.num_masses}-mass system.")

    # --- 1. Load Trained PINN Model ---
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please run train.py first.")
        return

    try:
        model = PINN(
            num_masses=args.num_masses,
            num_hidden_layers=args.hidden_layers,
            num_neurons=args.neurons
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # --- 2. Solve with Traditional ODE Solver ---
    logging.info("Solving system with traditional ODE solver (scipy.solve_ivp)...")
    
    # Initial Conditions for ODE Solver
    x_init = np.zeros(args.num_masses)
    x_init[0] = -args.x0
    v_init = np.zeros(args.num_masses)
    y0 = np.concatenate((x_init, v_init))  # Initial state vector
    
    # Time evaluation points
    t_eval = np.linspace(0, args.T_total, args.num_plot_points)
    
    try:
        ode_func = get_ode_system_func(args.num_masses, args.alpha)
        solution = solve_ivp(
            fun=ode_func,
            t_span=(0, args.T_total),
            y0=y0,
            t_eval=t_eval,
            method='RK45'  # High-precision solver
        )
        
        if not solution.success:
            logging.warning(f"ODE solver did not converge. Status: {solution.status}")
        
        ode_positions = solution.y[:args.num_masses]  # shape: (N, len(t_eval))
        logging.info("ODE solution obtained.")
    except Exception as e:
        logging.error(f"Failed to run ODE solver: {e}")
        return

    # --- 3. Get PINN Predictions ---
    logging.info("Generating PINN predictions...")
    try:
        t_eval_tensor = torch.tensor(t_eval.reshape(-1, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            pinn_pred = model(t_eval_tensor).cpu().numpy().T  # shape: (N, len(t_eval))
        logging.info("PINN predictions generated.")
    except Exception as e:
        logging.error(f"Failed to get PINN predictions: {e}")
        return
        
    # --- 4. Plot Comparison ---
    logging.info("Plotting comparison...")
    try:
        plot_save_path = os.path.join(args.plot_dir, "pinn_vs_ode_comparison.png")
        os.makedirs(args.plot_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 4 * args.num_masses))
        for i in range(args.num_masses):
            plt.subplot(args.num_masses, 1, i + 1)
            plt.plot(t_eval, ode_positions[i], 'k-', label=f'ODE Solver x{i+1}(t)', linewidth=2)
            plt.plot(t_eval, pinn_pred[i], 'r--', label=f'PINN Prediction x{i+1}(t)', linewidth=2)
            plt.xlabel('Time t')
            plt.ylabel(f'Position x{i+1}(t)')
            plt.grid(True)
            if i == 0:
                plt.legend()

        plt.suptitle(f'PINN vs. ODE Solver for N={args.num_masses} Coupled Masses', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(plot_save_path)
        logging.info(f"Comparison plot saved to {plot_save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot comparison: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PINN model against an ODE solver.")
    
    # System parameters (should match training)
    parser.add_argument('--num_masses', type=int, default=3, help="Number of masses (N)")
    parser.add_argument('--alpha', type=float, default=1.0, help="System constant alpha = k/m")
    parser.add_argument('--T_total', type=float, default=10.0, help="Total time duration (T)")
    parser.add_argument('--x0', type=float, default=1.0, help="Initial displacement of the first mass")
    
    # Model parameters (should match training)
    parser.add_argument('--hidden_layers', type=int, default=4, help="Number of hidden layers")
    parser.add_argument('--neurons', type=int, default=64, help="Number of neurons per hidden layer")

    # Evaluation parameters
    parser.add_argument('--num_plot_points', type=int, default=1000, help="Number of points to plot for comparison")

    # Logging and saving
    parser.add_argument('--log_file', type=str, default='logs/pinn_eval.log', help="Path to log file")
    parser.add_argument('--model_dir', type=str, default='models', help="Directory to load the trained model from")
    parser.add_argument('--model_name', type=str, default='pinn_model.pth', help="File name of the trained model")
    parser.add_argument('--plot_dir', type=str, default='plots', help="Directory to save comparison plots")

    args = parser.parse_args()
    main(args)
