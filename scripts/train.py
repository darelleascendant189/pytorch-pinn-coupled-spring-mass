"""
Main training script for the PINN model.
"""

import torch
import torch.optim as optim
import argparse
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure src directory is in path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import PINN
from src.data import generate_training_data
from src.physics_loss import physics_informed_loss, initial_condition_loss
from src.utils import setup_logging, get_device

def main(args):
    """
    Main training loop.
    """
    # Setup
    setup_logging(args.log_file)
    device = get_device()

    # Log hyperparameters
    logging.info("Starting training with hyperparameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Get data
    t_physics, t_ic, ic_data, t_val = generate_training_data(
        args.num_masses, args.T_total, args.x0,
        args.num_train_physics, args.num_train_ic, args.num_val,
        device
    )
    
    # Initialize model and optimizer
    model = PINN(
        num_masses=args.num_masses, 
        num_hidden_layers=args.hidden_layers, 
        num_neurons=args.neurons
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # Training loop
    loss_history = []
    val_loss_history = []
    epochs_log = []

    logging.info("Starting training loop...")
    pbar = tqdm(range(args.num_epochs), file=sys.stdout)
    for epoch in pbar:
        try:
            model.train()

            # 1. Physics Loss (on collocation points)
            loss_phys = physics_informed_loss(model, t_physics, args.alpha)
            
            # 2. Initial Condition Loss (at t=0)
            loss_ic = initial_condition_loss(model, t_ic, ic_data)
            
            # Total loss
            loss = args.w_physics * loss_phys + args.w_ic * loss_ic

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())

            # Validation
            if (epoch + 1) % args.log_freq == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_phys = physics_informed_loss(model, t_val, args.alpha)
                
                val_loss_item = val_loss_phys.item()
                val_loss_history.append(val_loss_item)
                epochs_log.append(epoch + 1)
                
                pbar.set_description(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Train Loss: {loss.item():.4e} | "
                    f"Val Loss: {val_loss_item:.4e}"
                )
                if (epoch + 1) % (args.log_freq * 10) == 0:
                     logging.info(
                        f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"Train Loss: {loss.item():.4e} (Physics: {loss_phys.item():.4e}, IC: {loss_ic.item():.4e}) | "
                        f"Val Loss: {val_loss_item:.4e}"
                    )

        except Exception as e:
            logging.error(f"Error in training loop at epoch {epoch + 1}: {e}")
            break

    logging.info("Training complete.")

    # Save model
    model_save_path = os.path.join(args.model_dir, args.model_name)
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # Plot and save loss curve
    try:
        plot_save_path = os.path.join(args.plot_dir, "loss_curve.png")
        os.makedirs(args.plot_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.num_epochs + 1), loss_history, label='Training Loss')
        plt.plot(epochs_log, val_loss_history, 'ro-', label='Validation Loss (Physics)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        logging.info(f"Loss curve plot saved to {plot_save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot loss curve: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN for a coupled spring-mass system.")
    
    # System parameters
    parser.add_argument('--num_masses', type=int, default=3, help="Number of masses (N)")
    parser.add_argument('--alpha', type=float, default=1.0, help="System constant alpha = k/m")
    parser.add_argument('--T_total', type=float, default=10.0, help="Total time duration (T)")
    parser.add_argument('--x0', type=float, default=1.0, help="Initial displacement of the first mass")
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10000, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument('--num_train_physics', type=int, default=5000, help="Number of physics collocation points")
    parser.add_argument('--num_train_ic', type=int, default=1000, help="Number of initial condition points (at t=0)")
    parser.add_argument('--num_val', type=int, default=1000, help="Number of validation points")
    parser.add_argument('--w_physics', type=float, default=1.0, help="Weight for the physics loss")
    parser.add_argument('--w_ic', type=float, default=10.0, help="Weight for the initial condition loss")
    parser.add_argument('--scheduler_step', type=int, default=2000, help="Step size for LR scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help="Gamma for LR scheduler")

    # Model parameters
    parser.add_argument('--hidden_layers', type=int, default=4, help="Number of hidden layers")
    parser.add_argument('--neurons', type=int, default=64, help="Number of neurons per hidden layer")
    
    # Logging and saving
    parser.add_argument('--log_freq', type=int, default=100, help="Frequency of logging validation loss")
    parser.add_argument('--log_file', type=str, default='logs/pinn_train.log', help="Path to log file")
    parser.add_argument('--model_dir', type=str, default='models', help="Directory to save the trained model")
    parser.add_argument('--model_name', type=str, default='pinn_model.pth', help="File name for the trained model")
    parser.add_argument('--plot_dir', type=str, default='plots', help="Directory to save plots")

    args = parser.parse_args()
    main(args)
