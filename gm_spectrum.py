"""
gm_spectrum.py
Test script for generating a 2D Gaussian Random Field with a Garrett-Munk-like spectrum.
Used to test the initial condition for the shallow water model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def generate_2d_gm_grf(n_points, L, k_star, slope=2.0):
    """
    Generate a 2D Gaussian Random Field with true complex Gaussian draws 
    for an isotropic Garrett-Munk-like spectrum.
    """
    # Generate 2D wavenumber grids
    kx = np.fft.fftfreq(n_points, d=L/n_points)
    ky = np.fft.rfftfreq(n_points, d=L/n_points)
    
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)
    
    # Define the theoretical 1D Energy Spectrum E(k)
    E_K = 1.0 / (k_star**2 + K_mag**2)**(slope / 2.0)
    E_K[0, 0] = 0.0  # Zero mean
    
    # Convert 1D Energy to 2D Power Density
    # P(k) = E(k) / (2 * pi * k)
    P_K = np.zeros_like(E_K)
    mask = K_mag > 0 
    P_K[mask] = E_K[mask] / (2.0 * np.pi * K_mag[mask])
    
    # True Gaussian Complex Noise
    noise_real = np.random.normal(0, 1, size=P_K.shape)
    noise_imag = np.random.normal(0, 1, size=P_K.shape)
    complex_noise = (noise_real + 1j * noise_imag) / np.sqrt(2.0)
    
    # Scale the complex Gaussian noise by the 2D amplitude spectrum
    complex_fourier_coeffs = np.sqrt(P_K) * complex_noise
    
    # Inverse 2D Fourier Transform
    spatial_field = np.fft.irfft2(complex_fourier_coeffs, s=(n_points, n_points))
    
    # Normalize to zero mean and unit variance
    spatial_field = (spatial_field - np.mean(spatial_field)) / np.std(spatial_field)
    
    return KX, KY, P_K, spatial_field

if __name__ == "__main__":
    # --- Parameters ---
    n_points = 512
    L = 1000.0       
    k_star = 0.02    
    slope = 2.0      

    # Generate the field
    KX, KY, P_K, gm_grf_2d = generate_2d_gm_grf(n_points, L, k_star, slope)

    # Compute the Empirical 2D Power Spectrum
    fft_2d = np.fft.fft2(gm_grf_2d)
    power_2d = np.abs(fft_2d)**2

    # Generate the full 2D wavenumber grids for binning
    kx_full = np.fft.fftfreq(n_points, d=L/n_points)
    KX_full, KY_full = np.meshgrid(kx_full, kx_full, indexing='ij')
    K_mag_full = np.sqrt(KX_full**2 + KY_full**2)

    # Calculate 1D Integrated Energy Spectrum E(k)
    K_flat = K_mag_full.flatten()
    power_flat = power_2d.flatten()

    # Define radial bins
    fundamental_freq = 1.0 / L
    nyquist_freq = (n_points / 2) * fundamental_freq
    k_bins = np.linspace(fundamental_freq, nyquist_freq, n_points // 2)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Digitize groups the 2D data into the 1D radial bins
    bin_indices = np.digitize(K_flat, k_bins)

    # Calculate the mean power in each ring (This is the 2D density P(k))
    radial_mean_power = np.array([power_flat[bin_indices == i].mean() for i in range(1, len(k_bins))])
    
    # Multiply by the shell circumference to get the 1D Integrated Energy E(k)
    integrated_1d_energy = radial_mean_power * (2 * np.pi * k_centers)

    # Theoretical curve for comparison
    theoretical_1d_energy = 1.0 / (k_star**2 + k_centers**2)**(slope / 2.0)

    # Scale to visually match the empirical data
    scaling_factor = np.median(integrated_1d_energy / theoretical_1d_energy)
    theoretical_1d_energy *= scaling_factor

    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Spatial Field
    im1 = ax1.imshow(gm_grf_2d, extent=[0, L, 0, L], origin='lower', cmap='RdBu_r')
    ax1.set_title("2D Spatial Field")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, label="Normalized Amplitude", shrink=0.8)
    
    # 2D Power Spectrum 
    fft_shifted = np.fft.fftshift(power_2d)
    kx_shifted = np.fft.fftshift(kx_full)
    im2 = ax2.imshow(fft_shifted, extent=[kx_shifted.min(), kx_shifted.max(), kx_shifted.min(), kx_shifted.max()],
                     origin='lower', cmap='viridis', norm=LogNorm(vmin=fft_shifted.max()*1e-6))
    ax2.set_title("2D Power Spectrum Map")
    ax2.set_xlabel("$k_x$")
    ax2.set_ylabel("$k_y$")
    fig.colorbar(im2, ax=ax2, label="Power", shrink=0.8)

    # 1D Integrated Energy Spectrum E(k)
    ax3.loglog(k_centers, integrated_1d_energy, 'o', color='royalblue', alpha=0.5, markersize=4, label='Empirical 1D $E(k)$')
    ax3.loglog(k_centers, theoretical_1d_energy, color='darkorange', linewidth=2, label=f'Theoretical $E(k)$ (Slope = -{slope})')
    ax3.axvline(k_star, color='grey', linestyle='--', label='$k_*$ (Roll-off)')

    ax3.set_title("1D Integrated Energy Spectrum")
    ax3.set_xlabel("Wavenumber Magnitude $|k|$")
    ax3.set_ylabel("Integrated Energy $E(k)$")
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()