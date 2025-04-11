import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# definerer radius
r = np.linspace(0, 20, 1000)

# simplifisert radiell sannsynlighetsfordeling
P_1s = 4 * r**2 * np.exp(-2*r)
P_2s = 4 * r**2 * (1 - r/2)**2 * np.exp(-r)
P_3s = 4 * r**2 * (1 - 2*r/3 + 2*(r**2)/27)**2 * np.exp(-2*r/3)

P_1s /= np.trapezoid(P_1s, r)
P_2s /= np.trapezoid(P_2s, r)
P_3s /= np.trapezoid(P_3s, r)

# Kumulativ sannsynlighet
cum_1s = cumulative_trapezoid(P_1s, r, initial=0)
cum_2s = cumulative_trapezoid(P_2s, r, initial=0)
cum_3s = cumulative_trapezoid(P_3s, r, initial=0)

# Finne r hvor kumulativ sannsynlighet er 90%
r_90_1s = r[np.searchsorted(cum_1s, 0.9)]
r_90_2s = r[np.searchsorted(cum_2s, 0.9)]
r_90_3s = r[np.searchsorted(cum_3s, 0.9)]

print(f'r_90 for 1s: {r_90_1s}')
print(f'r_90 for 2s: {r_90_2s}')
print(f'r_90 for 3s: {r_90_3s}')

# Plotter
plt.figure(figsize=(10, 6))
plt.plot(r, P_1s, label='1s', color='purple')
plt.plot(r, P_2s, label='2s', color='green')
plt.plot(r, P_3s, label='3s', color='red')

# Plotter linjer for å vise grensa hvor det er 90% sannsynlig å finne elektron innafor
plt.axvline(r_90_1s, color='purple', linestyle='--', label='90% radius 1s')
plt.axvline(r_90_2s, color='green', linestyle='--', label='90% radius 2s')
plt.axvline(r_90_3s, color='red', linestyle='--', label='90% radius 3s')

plt.xlabel('Avstand fra kjernen (r) [Bohr-radius]')
plt.ylabel('Sannsynlighet for å finne elektron (R²·r²)')
plt.title('Radiell sannsynlighetsfordeling for s-orbitaler til hydrogen')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
