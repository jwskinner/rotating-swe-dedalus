# Shallow Water Equation Models 
### Jack W. Skinner, Caltech (2025)

This code implements the 2D rotating non-dimensional shallow water model in the doubly periodic domain using **[Dedalus 3](https://dedalus-project.org/)**. 
We initialize the simulations with specific normal modes of the system, such as **geostrophically balanced flows** or propagating **inertia-gravity waves**.

The model generates an initial condition based on the **Garrett-Munk spectrum** with excited geostrophic or inertia-gravity wave modes, evolves the full nonlinear equations using the pseudo-spectral method, and outputs snapshots, diagnostics, and visualizations.

---

## Governing Equations
The model solves the **non-dimensional rotating shallow water equations** (SWEs) that describe the evolution of a thin layer of fluid of height $h$ with horizontal velocity $\mathbf{u} = (u,v)$ under the influence of rotation and gravity.

#### Momentum Equation 
$$\frac{\partial \mathbf{u}}{\partial t} + \frac{1}{\text{Ro}} \, \hat{\mathbf{z}} \times \mathbf{u} + \frac{1}{\text{Fr}^2} \nabla h + \nu_{\mathfrak{p}} \nabla^\mathfrak{p} \mathbf{u} = - (\mathbf{u} \cdot \nabla) \mathbf{u}$$

#### Continuity Equation 
$$\frac{\partial h}{\partial t} + \nabla \cdot \mathbf{u} + \nu_\mathfrak{p} \nabla^\mathfrak{p} h = - \nabla \cdot (h \mathbf{u})$$

#### Diagnostic Equations 
- **Relative Vorticity**: 

$$\zeta = \nabla \cdot (\hat{\mathbf{z}} \times \mathbf{u})$$

- **Divergence**: 

$$\delta = \nabla \cdot \mathbf{u}$$

Where:
* $\mathbf{u} = (u, v)$: horizontal velocity vector.
* $h$: perturbation height of the fluid from its mean height value.
* $\text{Ro}$: The **Rossby number**, representing the ratio of inertial to Coriolis forces.
* $\text{Fr}$: The **Froude number**, representing the ratio of flow speed to gravity wave speed.
* $L_d = \text{Ro}/\text{Fr}$: The **Rossby deformation radius**, representing the characteristic horizontal scale of the flow.
* $\nu$: The coefficient for hyperviscosity ($\nabla^{\mathfrak{p}}$ with $\mathfrak{p} = 8$ denoting the 8th-order hyperviscosity operator) used.

---

## Normal Modes of the System

The linearized shallow water equations support three solutions, or "normal modes," for each horizontal wavenumber vector $\mathbf{k} = (k, l)$. The initial conditions of the model are constructed by projecting a specified energy spectrum onto one of these modes.

### 1. Geostrophic Mode (Balanced Flow)
This is a **stationary** ($\omega=0$) and **non-divergent** flow where the Coriolis force precisely balances the pressure gradient force. 
The eigenvector in spectral space is:

$$(\hat{u}, \hat{v}, \hat{h}) \propto \left(-\frac{il}{\text{Fr}^2}, \frac{ik}{\text{Fr}^2}, \frac{1}{\text{Ro}}\right)$$

### 2. Inertia-Gravity Waves (IGW)
There are two IGW modes, propagating in opposite directions, with frequency given by the dispersion relation:

$$\omega(\mathbf{k}) = \pm \sqrt{\frac{1}{\text{Ro}^2} + \frac{|\mathbf{k}|^2}{\text{Fr}^2}}$$

The corresponding eigenvector is:

$$(\hat{u}, \hat{v}, \hat{h}) \propto \left(\frac{il}{\text{Ro}} + k\omega, -\frac{ik}{\text{Ro}} + l\omega, |\mathbf{k}|^2\right)$$

---

## Model Setup and Initialization

### Domain and Grid
The simulation domain is a doubly periodic Cartesian box. Its size, $L_x \times L_y$, is automatically configured based on the dominant physical scale of the normal mode chosen but can be adjusted in the `Config` class:

* For **balanced flow**, the domain is sized to 20 times the Rossby deformation radius, $L_d$.
* For **inertia-gravity waves**, the domain is also sized to 20 times the Rossby deformation radius, but can also be scaled by the IGW wavelength. 

### The Garrett-Munk (GM) Spectrum Target
The initial condition energy is distributed across wavenumbers following a 2D isotropic Garrett-Munk (GM) theoretical spectrum model. The 2D spectral power $P_{2D}(K)$ as a function of the isotropic wavenumber magnitude $K = |\mathbf{k}|$ is defined as:

$$P_{2D}(K) = \frac{1}{(k_*^2 + K^2)^{p/2}}$$

Where:
* $k_*$: The characteristic roll-off wavenumber, which controls the transition from the flat low-wavenumber plateau to the inertial subrange which follows a $k^{-2}$ power law.
* $p$: The 2D spectral slope, which dictates the rate of energy cascade at high wavenumbers. A 2D slope of $p=3.0$ corresponds to a 1D isotropic spectral slope of $K^{-2}$.

### Initialization Algorithm
The initial physical fields $u(x,y)$, $v(x,y)$, and $h(x,y)$ are generated in the `InitialConditions` class:

1. **Wavenumber Masking**: The algorithm generates a discrete 2D grid of angular wavenumbers $(k_x, k_y)$ and isolates all non-zero modes ($K > 0$).
2. **Eigenvector Projection**: The spatial projection eigenvectors $(\hat{u}, \hat{v}, \hat{h})$ are computed for every non-zero mode using either the geostrophic or wave relations above.
3. **Amplitude Scaling**: A scaling factor $\sigma$ is computed for each mode. This factor normalizes the kinetic energy weight of the chosen eigenvector and scales it to match the GM power spectrum $P_{2D}(K)$ multiplied by a base amplitude $A$.
4. **Spectral Tapering**: A smooth exponential taper is then applied to the amplitude based on a high-wavenumber cutoff $k_c$.
5. **Stochastic Realization**: Each mode is multiplied by normally distributed, complex random phases: $\frac{1}{\sqrt{2}}(\mathcal{N}(0,1) + i\mathcal{N}(0,1))$.
6. **Physical Transformation**: The complex spectral fields are transformed back to physical space to yield the initial $u, v, h$ fields passed to the Dedalus solver.

---

## Configuration Parameters

All key simulation parameters are controlled via the `Config` dataclass at the top of the script.

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `sim_number` | Label for the simulation run (for parallel jobs) | `1` |
| `ic_mode` | Initial condition mode (`balanced` or `wave`) | `balanced` |
| `Nx`, `Ny` | Grid resolution | `256` |
| `Ro`, `Fr` | Rossby and Froude numbers | `0.1`, `0.1` |
| `Re` | Hyper-Reynolds number for 8th-order hyper-dissipation | `1e13` |
| `k_c` | High wavenumber cutoff for exponential taper | `30.0` |
| `amp` | Amplitude of the initial spectral energy | `0.1` |
| `k_*` (`k_star`) | Characteristic GM roll-off wavenumber | `2.0` |
| `slope` | High-k 2D spectral slope | `3.0` |
| `dt` | Timestep for the solver | `0.01` |
| `out_freq` | Frequency of output for snapshots and frames | `0.2` |
| `sim_time` | Total simulation time (in non-dimensional units) | `201.0` |

---

## Outputs and Visualization 

The model creates output directories in the `outputs/` folder based on the run configuration. It produces two types of output:

1.  **Snapshots**: HDF5 files containing `height`, `u_vec` (velocity vector), `div` (divergence), `zeta` (vorticity), and `total_energy`. These are saved in the `snapshot_dir` (`./outputs/snapshots/{tag}`).
2.  **Frame Images**: Plots of the flow and spectra at regular intervals, saved in `frame_dir` (`./outputs/frames/{tag}`).
---

## Running the Model

```bash
# Run with defaults (sim_number = 1, balanced mode)
python sw_model.py

# Run a wave simulation with sim_number = 2
python sw_model.py 2 wave

# Run a balanced simulation with sim_number = 3
python sw_model.py 3 balanced