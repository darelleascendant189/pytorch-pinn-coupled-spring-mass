"""
Defines the physics-informed loss functions for the spring-mass system.
"""

import torch
import torch.nn as nn

def compute_derivatives(model: nn.Module, t: torch.Tensor, order: int = 1):
    """
    Efficiently computes derivatives of the model output w.r.t. input time.
    """
    t.requires_grad_(True)
    x = model(t)  # [batch_size, num_masses]
    
    # Compute first derivative (velocity)
    x_t = torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]  # [batch_size, num_masses]

    if order == 1:
        return x, x_t

    # Compute second derivative (acceleration)
    x_tt = torch.autograd.grad(
        x_t, t,
        grad_outputs=torch.ones_like(x_t),
        create_graph=True
    )[0]  # [batch_size, num_masses]
    
    return x, x_t, x_tt

def physics_informed_loss(model: nn.Module, t_physics: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the physics-informed loss (ODE residual).

    Args:
        model (nn.Module): The PINN model.
        t_physics (torch.Tensor): Collocation points for physics loss.
        alpha (float): The system constant (k/m).

    Returns:
        torch.Tensor: The mean squared error of the ODE residuals.
    """
    # Compute positions, velocities, and accelerations
    x, _, x_tt = compute_derivatives(model, t_physics, order=2)
    
    num_masses = x.shape[1]
    residuals = torch.zeros_like(x_tt)
    
    # Calculate residuals based on the ODEs
    for i in range(num_masses):
        # x_left is x_{i-1}
        x_left = x[:, i - 1] if i > 0 else torch.zeros_like(x[:, i])
        
        # x_right is x_{i+1}
        x_right = x[:, i + 1] if i < num_masses - 1 else torch.zeros_like(x[:, i])
        
        # ODE: x_tt[i] = alpha * (x_right - 2*x[i] + x_left)
        # Residual: f[i] = x_tt[i] - alpha * (x_right - 2*x[i] + x_left)
        residuals[:, i] = x_tt[:, i] - alpha * (x_right - 2 * x[:, i] + x_left)

    loss_phys = torch.mean(residuals**2)
    return loss_phys

def initial_condition_loss(model: nn.Module, t_ic: torch.Tensor, ic_data: dict) -> torch.Tensor:
    """
    Computes the initial condition loss (position and velocity at t=0).

    Args:
        model (nn.Module): The PINN model.
        t_ic (torch.Tensor): Time points for initial conditions (all t=0).
        ic_data (dict): Dictionary with target initial positions ('pos') and velocities ('vel').

    Returns:
        torch.Tensor: The combined mean squared error for initial positions and velocities.
    """
    # Compute positions and velocities at t=0
    x_ic, x_t_ic = compute_derivatives(model, t_ic, order=1)
    
    # Target initial positions and velocities
    target_x0 = ic_data['pos']
    target_v0 = ic_data['vel']
    
    # Position loss at t=0
    loss_pos = torch.mean((x_ic - target_x0)**2)
    
    # Velocity loss at t=0
    loss_vel = torch.mean((x_t_ic - target_v0)**2)
    
    loss_ic = loss_pos + loss_vel
    return loss_ic
