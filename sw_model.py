"""
sw_model.py
JWS, Non-dimensional Rotating Shallow Water Model using Dedalus
Caltech, 2025

Model uses the IVP solver from Dedalus to solve the Non-Dim SWE with two modes:
  - 'balanced' : geostrophically-balanced (vortex) initial conditions
  - 'wave'     : inertia-gravity wave initial conditions

How to run:
    python sw_model.py [sim_number] [balanced|wave]

E.g.,
    python sw_model.py 1 balanced
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xrft

# Set the number of threads to 1 for Dedalus
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Configuration for the Shallow Water Model."""
    sim_number: int = 1
    ic_mode: str = 'balanced'   # 'balanced' or 'wave'
    
    # grid resolution
    Nx: int = 256
    Ny: int = 256
    
    # flow params
    Ro: float = 0.1   # Rossby number
    Fr: float = 0.1   # Froude number
    Re: float = 1e13  # Reynolds number (Hyper-reynolds number for 8th order hyper-dissipation)
    Ld: float = Ro / Fr # Deformation radius

    # domain params
    Lx: float = 20 * Ld 
    Ly: float = 20 * Ld
    dx: float = Lx / Nx
    dy: float = Ly / Ny
    
    # Spectral parameters for initial condition
    # All in units of Angular Wavenumber 'k'
    # Define a wavenumber band to excite
    k_c: float = 30.0                  # high wavenumber cutoff for exponential taper
    amp: float = 1e-2                  # Amplitude of the initial condition (5e-2 for balanced, 1e-2 for wave to prevent nonlinear steepening)

    # GM spectrum model
    k_star: float = 1.0                 # Characteristic GM roll-off wavenumber
    slope: float = 3.0                  # High-k 2D spectral slope (1D slope of k^-2)

    # Time-stepping parameters (defaults set conditionally in __post_init__ depending on IC mode)
    dt: float | None = None
    out_freq: float | None = None
    sim_time: float | None = None

    frame_dir: Path = field(init=False)
    snapshot_dir: Path = field(init=False)

    def __post_init__(self):

        # Simulation parameters (balanced)
        if self.ic_mode == 'balanced':
            print("Using balanced IC setup")
            self.dt: float = 0.09           # Time step (100th of an eddy turnover time at the deformation radius)
            self.out_freq: float = 0.9      # Output frequency (every 0.1 eddy turnover)
            self.sim_time: float = 900.0    # Simulation time (100 turnover times)
        
        # Simulation parameters (wave)
        elif self.ic_mode == 'wave':
            T_gw_dx = self.dx * self.Fr       # Grid-scale gravity wave crossing time
            T_inertial = 2 * np.pi * self.Ro  # Inertial period of gravity wave
            print(f"[IC:wave] setup | Grid-scale gravity wave crossing time={T_gw_dx:.5f} | Inertial period={T_inertial:.3f}")

            self.dt: float = 0.0007          # 1/10th of the grid-scale gravity wave crossing time
            self.out_freq: float = 0.06      # 1/10th of the inertial period to capture wave oscillations
            self.sim_time: float = 61.0      # 100 inertial periods (+1 to ensure last frame is output)

        # setup IO
        tag = f"{self.ic_mode}_{self.Nx}_{self.sim_number}"
        self.frame_dir = Path(f"./outputs/outputs_test/frames/{tag}")
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = Path(f"./outputs/outputs_test/snapshots/{tag}")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)


# ── Spectral Functions ──────────────────────────────────────────────────────────

def compute_1d_spectrum(field_data: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the 1D isotropic power spectrum.
    Returns: (k, power_1d) where k is the angular wavenumber.
    """
    da = xr.DataArray(field_data, dims=['x', 'y'], coords={'x': x, 'y': y})
    iso = xrft.isotropic_power_spectrum(da, dim=['x', 'y'], window=None, detrend=None, truncate=True).compute()
    k = 2 * np.pi * iso.freq_r.values 
    power_1d = iso.values
    return k, power_1d

def gm_reference(k: np.ndarray, cfg: Config) -> np.ndarray:
    """Generates the Garrett-Munk theoretical 1D spectrum as a function of k."""
    k_norm = k / cfg.k_star
    gm_power_2d = (1.0 + k_norm**2) ** (-cfg.slope / 2.0)
    return k * gm_power_2d

# ── Mode Setup & Projections ──────────────────────────────────────────

def project_balanced(k, l, Ro, Fr):
            """Geostrophically balanced vortex mode."""
            u_hat = -1j * l / Fr**2
            v_hat =  1j * k / Fr**2
            h_hat =  np.full_like(k, 1 / Ro)
            return u_hat, v_hat, h_hat

def project_wave(k, l, Ro, Fr):
    """Inertia-gravity wave mode."""
    K_sq = k**2 + l**2
    omega = np.sqrt(Ro**-2 + K_sq / Fr**2)  # Wave dispersion relation
    
    u_hat =  1j * l / Ro + k * omega
    v_hat = -1j * k / Ro + l * omega
    h_hat =  K_sq
    return u_hat, v_hat, h_hat

# ── Initial Conditions ────────────────────────────────────────────────────────

class InitialConditions:
    """Generates initial conditions in spectral space and transforms to physical space."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        Ro, Fr, Ld = cfg.Ro, cfg.Fr, cfg.Ld
        
        # Apply the selected mode
        if cfg.ic_mode == 'balanced':
            proj_fn = lambda k, l: project_balanced(k, l, Ro, Fr)
            
        elif cfg.ic_mode == 'wave':
            #cfg.Lx = cfg.Ly = 20 * (2 * np.pi * Ld**2) # use wave scale if we like (optional)
            proj_fn = lambda k, l: project_wave(k, l, Ro, Fr)
            
        else:
            raise ValueError(f"Unknown ic_mode: '{cfg.ic_mode}'. Use 'balanced' or 'wave'.")

        self.x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx)
        self.y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny)

        KX, KY = np.meshgrid(
            2 * np.pi * np.fft.fftfreq(cfg.Nx, cfg.dx),
            2 * np.pi * np.fft.fftfreq(cfg.Ny, cfg.Ly / cfg.Ny),
            indexing='ij'
        )
        K = np.sqrt(KX**2 + KY**2)

        print(f"[IC:{cfg.ic_mode}] Ld={Ld:.3f} CFL-safe dt={0.1 * cfg.dx * Fr:.5f}")

        # Generate IC fields in spectral space
        mask = K > 0 # remove zero mode from u, v, h
        pu, pv, ph = proj_fn(KX[mask], KY[mask])
        
        k_norm = K[mask] / self.cfg.k_star
        gm_power_2d = (1.0 + k_norm**2) ** (-self.cfg.slope / 2.0)

        # Base spectral amplitude from gm spectrum
        ke_weight = np.maximum(np.abs(pu)**2 + np.abs(pv)**2, 1e-30)
        sigma = cfg.amp * (cfg.Nx * cfg.Ny) * np.sqrt(gm_power_2d / ke_weight)

        # Apply a smooth exponential taper to sigma
        sigma *= np.exp(- 2.0 * (K[mask] / (cfg.k_c)) ** 4.0)

        # Weight a Gaussian random field by sigma
        coeffs = sigma * (np.random.standard_normal(mask.sum()) + 1j * np.random.standard_normal(mask.sum())) / np.sqrt(2)

        Fu, Fv, Fh = np.zeros_like(K, dtype=complex), np.zeros_like(K, dtype=complex), np.zeros_like(K, dtype=complex)
        Fu[mask], Fv[mask], Fh[mask] = pu * coeffs, pv * coeffs, ph * coeffs

        ri = lambda F: np.real(np.fft.ifft2(F))
        self.u, self.v, self.h = ri(Fu), ri(Fv), ri(Fh)
        self.zeta = ri(1j * (KX * Fv - KY * Fu))
        self.div = ri(1j * (KX * Fu + KY * Fv))

    def plot_ic_spectrum(self):
        """Plot the actual IC spectrum vs the target GM reference."""
        k, iso_u = compute_1d_spectrum(self.u, self.x, self.y)
        _, iso_v = compute_1d_spectrum(self.v, self.x, self.y)
        _, iso_h_1d = compute_1d_spectrum(self.h, self.x, self.y)
        
        ke = 0.5 * (iso_u + iso_v)
        ref_1d = gm_reference(k, self.cfg)

        scale_factor = ke[0] / ref_1d[0] # adjust amp of model fit so it matched the IC
        ref_1d_scaled = ref_1d * scale_factor

        k2_ref = 1e2 * ke[0] * (k / k[0]) ** -2

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(k, ke, 'b-', lw=1.5, label='KE (Empirical)')
        ax.loglog(k, iso_h_1d, 'g-', lw=1.5, label=r'$h$')
        ax.loglog(k, ref_1d_scaled, color='darkorange', ls='-', lw=2, label=f'GM Model (p={self.cfg.slope})')
        ax.axvline(self.cfg.k_star, color='grey', linestyle=':', label='$k_*$ (Roll-off)')
        ax.loglog(k, k2_ref, 'k--', lw=1, alpha=0.6, label=r'$k^{-2}$')
        
        # label the Nyquist limit (for tuning the taper)
        k_nyquist = np.pi / (cfg.Lx / cfg.Nx)
        ax.axvline(k_nyquist, color='black', linestyle='-', alpha=0.3, label='Nyquist Limit')
        ax.set(xlabel=r'Angular Wavenumber $k$', ylabel='1D Spectral Power', title='IC Isotropic Power Spectrum')
        ax.set_ylim(bottom=1e-9)
        ax.legend()
        plt.tight_layout()
        plt.savefig('ic_spectrum_check.png', dpi=150, bbox_inches='tight')
        plt.close()


# ── Dedalus Solver ────────────────────────────────────────────────────────────────────

class ShallowWaterSolver:
    """Manages the Dedalus IVP solver for the shallow water equations."""
    
    def __init__(self, ic: InitialConditions, cfg: Config):
        self.ic = ic
        self.cfg = cfg
        self.frame_count = 1
        self._build()

    def _build(self):
        coords = d3.CartesianCoordinates('x', 'y')
        dist = d3.Distributor(coords, dtype=np.float64)
        xb = d3.RealFourier(coords['x'], cfg.Nx, bounds=(-cfg.Lx / 2, cfg.Lx / 2), dealias=1.5)
        yb = d3.RealFourier(coords['y'], cfg.Ny, bounds=(-cfg.Ly / 2, cfg.Ly / 2), dealias=1.5)

        self.u_vec = dist.VectorField(coords, name='u_vec', bases=(xb, yb))
        self.h = dist.Field(name='h', bases=(xb, yb))
        self.zeta = dist.Field(name='zeta', bases=(xb, yb))
        self.delta = dist.Field(name='delta', bases=(xb, yb))

        self.u_vec['g'][0], self.u_vec['g'][1] = self.ic.u, self.ic.v
        self.h['g'], self.zeta['g'], self.delta['g'] = self.ic.h, self.ic.zeta, self.ic.div

        Ro, Fr, nu = self.cfg.Ro, self.cfg.Fr, 1 / self.cfg.Re
        zcross = lambda A: d3.skew(A)
        
        prob = d3.IVP([self.u_vec, self.h, self.zeta, self.delta], namespace=locals())
        prob.add_equation("dt(u_vec) + (1/Ro)*zcross(u_vec) + (1/Fr**2)*grad(h) + nu*lap(lap(lap(lap(u_vec)))) = -u_vec@grad(u_vec)")
        prob.add_equation("dt(h) + div(u_vec) + nu*lap(lap(lap(lap(h)))) = -div(h*u_vec)")
        prob.add_equation("zeta - div(skew(u_vec)) = 0")
        prob.add_equation("delta - div(u_vec) = 0")

        self.solver = prob.build_solver(d3.RK443) # 4th order Runge-Kutta with 3rd order error estimate
        self.flow = d3.GlobalFlowProperty(self.solver, cadence=1)
        self.flow.add_property(np.sqrt(self.u_vec @ self.u_vec), name='speed')
        self.flow.add_property(d3.abs(self.zeta), name='abs_zeta')   # Absolute vorticity
        self.flow.add_property(d3.abs(self.delta), name='abs_delta') # Absolute divergence
        self.flow.add_property(self.h, name='height')                # Height

        snaps = self.solver.evaluator.add_file_handler(self.cfg.snapshot_dir, sim_dt=self.cfg.out_freq, max_writes=10_000)
        snaps.add_task(self.h, name='height', layout='g')
        snaps.add_task(self.u_vec, name='u_vec', layout='g')
        snaps.add_task(d3.div(self.u_vec), name='div')
        snaps.add_task(d3.div(d3.skew(self.u_vec)), name='zeta')
        snaps.add_task(d3.integ(0.5*(self.u_vec@self.u_vec) + self.h**2/Fr**2), name='total_energy')
        snaps.add_task(d3.lap(self.h), name='lap_h', layout='g') # output the laplacian of h

    def run(self):
        cfg = self.cfg
        dt = np.float64(cfg.dt)
        cadence = round(cfg.out_freq / cfg.dt)
        
        self.solver.stop_sim_time = cfg.sim_time
        self._save_frame(0.0)

        while self.solver.proceed:
            self.solver.step(dt)
            if self.solver.iteration % cadence == 0:
                t = self.solver.sim_time
                speed = self.flow.max('speed')
                max_z = self.flow.max('abs_zeta')
                max_d = self.flow.max('abs_delta')
                max_h = self.flow.max('height')
                cfl = speed * dt / (2 * cfg.Lx / cfg.Nx)
                t_adv = cfg.Ld / speed if speed > 1e-10 else 0.0
                # output string 
                print(f"iter={self.solver.iteration:6d} | "
                      f"t={t:7.3f} | "
                      f"CFL={cfl:6.3f} | "
                      f"Tadv={t_adv:6.3f} | "
                      f"Ro_eff={speed*cfg.Ro:6.3f} | "
                      f"Fr_eff={speed*cfg.Fr:6.3f} | "
                      f"max_ζ={max_z:7.3f} | "
                      f"max_δ={max_d:7.3f} | "
                      f"max_h={max_h:7.3f} |")
                self._save_frame(t)

    def _save_frame(self, t: float):
        u, v = self.u_vec['g'] if t > 0.0 else (self.ic.u, self.ic.v)
        h = self.h['g'] if t > 0.0 else self.ic.h
        zeta = self.zeta['g'] if t > 0.0 else self.ic.zeta
        delta = self.delta['g'] if t > 0.0 else self.ic.div

        xg = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, u.shape[0])
        yg = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, u.shape[1])
        
        fig = plot_diagnostics(u, v, h, zeta, delta, xg, yg, t, self.cfg)
        fig.savefig(self.cfg.frame_dir / f"frame_{self.frame_count:04d}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        self.frame_count += 1


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_diagnostics(u: np.ndarray, v: np.ndarray, h: np.ndarray, zeta: np.ndarray, delta: np.ndarray, 
                     x: np.ndarray, y: np.ndarray, t: float, cfg: Config) -> plt.Figure:
    """Generates a multi-panel plot with k^-2 and k^-5/3 reference lines."""
    k, iso_u = compute_1d_spectrum(u, x, y)
    _, iso_v = compute_1d_spectrum(v, x, y)
    _, iso_h_1d = compute_1d_spectrum(h, x, y)
    ke_1d = 0.5 * (iso_u + iso_v)

    fig, axs = plt.subplots(1, 5, figsize=(23, 4))
    kw = dict(extent=(x.min(), x.max(), y.min(), y.max()), aspect='equal', origin='lower', interpolation='nearest')

    # Spatial Plots
    plot_configs = [
        (axs[0], np.sqrt(u**2 + v**2), 'viridis', (0, 0.1), r'$|\mathbf{u}|$'),
        (axs[1], zeta, 'RdBu_r', (-1.0, 1.0), r'$\zeta$'),
        (axs[2], delta, 'RdBu_r', (-1.0, 1.0), r'$\delta$'),
        (axs[3], h, 'Blues_r', (-0.01, 0.01), r'${h}$'),
    ]

    for ax, data, cmap, (lo, hi), title in plot_configs:
        im = ax.imshow(data.T, cmap=cmap, vmin=lo, vmax=hi, **kw)
        ax.locator_params(axis='both', nbins=5)
        ax.set(xlabel='x', ylabel='y', title=title)
        fig.colorbar(im, ax=ax, shrink=0.6)

    # Spectral Plot
    ax_spec = axs[4]
    ax_spec.loglog(k, ke_1d, 'b-', lw=2, label='KE')
    ax_spec.loglog(k, iso_h_1d, 'g-', lw=2, label=r'${h}$')
    
    # --- Reference Lines ---
    # Anchored to the mid-range of the spectrum
    k_ref = k[(k > 0.1) & (k < 50)]
    if len(k_ref) > 0:
        
        # Reference point for scaling the lines
        anchor_k = k_ref[0]
        anchor_val = 1e2 * ke_1d[np.where(k == anchor_k)][0]
        
        # k^-2 line
        ax_spec.loglog(k_ref, anchor_val * (k_ref / anchor_k)**-2, 
                       color='k', ls='--', lw=1.5, label=r'$k^{-2}$')
        
        # k^-5/3 line
        ax_spec.loglog(k_ref, anchor_val * (k_ref / anchor_k)**(-5/3), 
                       color='k', ls=':', lw=1.5, label=r'$k^{-5/3}$')

    ax_spec.set(xlabel=r'$k$', ylabel='Power', title='Isotropic Power Spectrum')
    ax_spec.set_ylim(bottom=1e-12, top=ke_1d.max()*1e2)
    ax_spec.grid(True, which='both', ls=':', alpha=0.5)
    ax_spec.legend(loc='lower left', fontsize='small')

    fig.suptitle(f'Simulation {cfg.sim_number} ({cfg.ic_mode}) | t = {t:.3f}', fontsize=16)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotating Shallow Water Model")
    parser.add_argument("sim_number", type=int, nargs="?", default=1, help="Simulation number")
    parser.add_argument("ic_mode", type=str, nargs="?", default="balanced", choices=["balanced", "wave"], help="Initial condition mode")
    
    args = parser.parse_args()
    cfg = Config(sim_number=args.sim_number, ic_mode=args.ic_mode)
    
    ic = InitialConditions(cfg)
    ic.plot_ic_spectrum()
    
    solver = ShallowWaterSolver(ic, cfg)
    solver.run()