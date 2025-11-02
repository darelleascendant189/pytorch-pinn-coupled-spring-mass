# Physics-Informed Neural Network (PINN) for Coupled Spring-Mass System

This project demonstrates a powerful, modern approach to solving complex physics problems using deep learning. It features a Physics-Informed Neural Network (PINN) implemented in PyTorch to solve the system of ordinary differential equations (ODEs) governing an N-dimensional coupled spring-mass system. Unlike traditional neural networks that require large, pre-solved datasets for training, this PINN learns the solution by directly embedding the system's governing physical laws (Newton's 2nd Law) into its loss function. 

This repository, developed for the M.S. Deep Learning course (Spring 2025), provides a complete and modular implementation of this data-free differential equation solver.

## Features

* **Data-Free Solver:** Solves a system of N-dimensional ODEs using *no* ground-truth solution data.
* **Physics-Informed Loss:** Embeds the system's differential equations directly into the loss function.
* **Initial Condition Enforcement:** Uses a separate loss component to enforce initial positions and velocities.
* **Efficient Derivatives:** Uses vectorized `torch.autograd.grad` for fast and efficient computation of 1st and 2nd order time derivatives.
* **Configuration:** Uses `argparse` to easily configure system parameters (e.g., number of masses) and training hyperparameters.
* **Logging:** Logs all training and evaluation progress to both the console and timestamped log files in the `logs/` directory.

## Core Concepts & Techniques

* **Physics-Informed Neural Networks (PINNs):** Leveraging physical laws as a regularization term to constrain the solution space of a neural network.
* **Scientific Machine Learning (SciML):** Applying deep learning techniques to solve complex problems in science and engineering.
* **PyTorch Automatic Differentiation:** Using `autograd` to compute the derivatives ($\dot{x}$, $\ddot{x}$) required for the ODE residual.
* **Deep Feed-Forward Networks:** Using a simple FFN to act as a universal function approximator for the solution $x(t)$.

---

## How It Works

This project trains a neural network to solve a system of differential equations. The "magic" of a Physics-Informed Neural Network (PINN) is that it learns the solution *without* ever being shown a pre-solved example. Instead, it learns by trying to satisfy the physics equations directly.

This is achieved through a custom loss function that is a weighted sum of two components:
1.  **Data Loss ($L_{ic}$):** Enforces the known "facts" of the system (the initial conditions).
2.  **Physics Loss ($L_{physics}$):** Enforces the governing laws of physics (the differential equation).

$$
L_{\text{total}} = w_{\text{ic}} L_{\text{ic}} + w_{\text{physics}} L_{\text{physics}}
$$

The network, a simple feed-forward model, takes a single time value $t$ as input and outputs the $N$ positions of the masses, $\mathbf{x}(t) = [x_1(t), ..., x_N(t)]$. The optimizer's job is to find network weights $\theta$ that minimize this $L_{\text{total}}$.


### 1. The Physics (Residual) Loss: $L_{physics}$

This is the core idea of the PINN. It ensures the network's output "obeys the laws of physics" at any given time $t > 0$.

1.  **The Law of Physics:** The equation of motion for the $i$-th mass is:

    $$m\ddot{x}\_i = k(x_{i+1} - x_i) - k(x_i - x_{i-1})$$

    Letting $\alpha = k/m$ and rearranging, we get a "residual" equation $f_i(t)$ that must equal zero:

    $$f_i(t) = \ddot{x}\_i(t) - \alpha (x_{i+1}(t) - 2x_i(t) + x_{i-1}(t)) = 0$$

3.  **Calculating the Loss:**
    * We feed the network a batch of random time points $t_j$ (called **collocation points**).
    * For each $t_j$, the network predicts the positions $x_i(t_j)$.
    * Using PyTorch's **Automatic Differentiation** (`autograd`), we compute the second derivative of the network's output with respect to its input, giving us the acceleration $\ddot{x}_i(t_j)$.
    * We plug these predicted $x_i$ and $\ddot{x}_i$ values into the residual equation $f_i(t_j)$.
    * The loss $L_{physics}$ is the **Mean Squared Error** of these residuals. By minimizing this loss, we force the residuals toward zero, thus forcing the network's predictions to satisfy the differential equation.

    $$L_{\text{physics}} = \frac{1}{N_{\text{physics}}} \sum_{j} \sum_{i=1}^{N} \left( f_i(t_j) \right)^2$$

This logic is implemented in `src/physics_loss.py:physics_informed_loss`.

### 2. The Initial Condition (Data) Loss: $L_{ic}$

This loss component "pins" the solution, ensuring it starts at the correct state at $t=0$. This is the only "data" we use. We have two known conditions at $t=0$:

1.  **Initial Position:** All masses are at rest, except the first:
    $\mathbf{x}(0) = [-x_0, 0, ..., 0]$
2.  **Initial Velocity:** All masses are at rest:
    $\dot{\mathbf{x}}(0) = [0, 0, ..., 0]$

To calculate this loss, we feed a batch of $t=0$ points into the network:
* We get the network's position prediction $\mathbf{x}(0)$ and compute its MSE against the target positions.
* We use `autograd` (this time for the *first* derivative) to get the velocity prediction $\dot{\mathbf{x}}(0)$ and compute its MSE against the target velocities (all zeros).

$$
L_{\text{ic}} = \underbrace{\text{MSE}(\mathbf{x}(0), \mathbf{x}_{\text{target}})}_{\text{Position Loss}} + \underbrace{\text{MSE}(\dot{\mathbf{x}}(0), \dot{\mathbf{x}}_{\text{target}})}_{\text{Velocity Loss}}
$$

This logic is implemented in `src/physics_loss.py:initial_condition_loss`.

### 3. The Result: A Trained Solver

By minimizing the combined $L_{\text{total}}$, the optimizer finds a set of network weights that produce a function $\mathbf{x}(t)$ that both **starts at the right place** (by minimizing $L_{ic}$) and **evolves according to the correct physical laws** (by minimizing $L_{physics}$).

The fully trained network *is* the solution. We can then feed it any time $t$ in our domain and it will return the predicted positions of all masses.

---

## Project Structure

```
pytorch-pinn-coupled-spring-mass/
├── .gitignore            # Standard Python .gitignore
├── LICENSE               # MIT License file
├── README.md             # This readme file
├── requirements.txt      # Project dependencies (torch, numpy, etc.)
├── logs/
│   └── .gitkeep          # Directory for log files (e.g., pinn_train.log)
├── models/
│   └── .gitkeep          # Directory for saved model weights (e.g., pinn_model.pth)
├── plots/
│   └── .gitkeep          # Directory for output plots (loss_curve.png, etc.)
├── src/
│   ├── init.py           # Makes src a package
│   ├── data.py           # generate_training_data function
│   ├── model.py          # PINN class (the nn.Module)
│   ├── physics_loss.py   # physics_informed_loss and initial_condition_loss
│   └── utils.py          # Utility functions (e.g., setup_logging, get_device)
├── scripts/
│   ├── train.py          # Main script to train the PINN model
│   └── evaluate.py       # Script to evaluate the model against an ODE solver
└── run_project.ipynb     # A guide notebook to run the project step-by-step
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-pinn-coupled-spring-mass.git
    cd pytorch-pinn-coupled-spring-mass
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Train the Model:**
    Run the training script. You can customize parameters using command-line arguments.
    ```bash
    python scripts/train.py --num_epochs 10000 --num_masses 3 --w_ic 10.0
    ```
    * This will train the model, save weights to `models/pinn_model.pth`, and save a loss plot to `plots/loss_curve.png`.
    * Run `python scripts/train.py --help` to see all available options.

4.  **Evaluate the Model:**
    After training, run the evaluation script to compare the PINN's solution to a traditional ODE solver.
    ```bash
    python scripts/evaluate.py --num_masses 3
    ```
    * This loads the saved model and generates a comparison plot at `plots/pinn_vs_ode_comparison.png`.

5.  **Run with the Guide Notebook:**
    For a more guided, step-by-step experience, open and run the `run_project.ipynb` notebook in Jupyter.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
