# lab06_sir_sim.py
# Simulación SIR (partículas y autómata celular), promediado sobre Nexp,
# comparación con la solución ODE y exportación de animaciones (.gif / .mp4) y gráficas.
# Archivos generados en out_dir (por defecto: /mnt/data).
#
# Requisitos: numpy, matplotlib, imageio, scipy, pillow
# En entornos sin ffmpeg, se generarán GIF; si ffmpeg está disponible, también MP4.

import numpy as np
import matplotlib.pyplot as plt
import imageio, os
from scipy.integrate import solve_ivp
from scipy.signal import convolve2d
from PIL import Image

# ---------------------------
# Funciones auxiliares SIR
# ---------------------------
def sir_ode(t, y, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

def sir_theoretical(S0, I0, R0, beta, gamma, N, t_grid):
    sol = solve_ivp(lambda t, y: sir_ode(t, y, beta, gamma, N),
                    [t_grid[0], t_grid[-1]], [S0, I0, R0],
                    t_eval=t_grid, vectorized=False, rtol=1e-6)
    return sol.y.T  # shape (len(t_grid), 3)

# ---------------------------
# Utilidad para guardar frames con tamaño consistente
# ---------------------------
def save_frame_consistent(frames_list, fig, target_shape_holder):
    fig.canvas.draw()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches='tight')
    buf.seek(0)
    arr = imageio.v2.imread(buf)
    if target_shape_holder[0] is None:
        target_shape_holder[0] = arr.shape
        frames_list.append(arr)
    else:
        target = target_shape_holder[0]
        if arr.shape != target:
            pil = Image.fromarray(arr)
            pil = pil.resize((target[1], target[0]), Image.NEAREST)
            arr2 = np.array(pil)
            frames_list.append(arr2)
        else:
            frames_list.append(arr)

# ---------------------------
# Simulación de partículas móviles
# ---------------------------
def run_experiments_particle(Nexp, L, Ntotal, I0, vmax, r, beta, gamma, dt, T, base_seed, make_anim=True):
    all_SIR = []
    frames_example = None
    rng0 = np.random.RandomState(base_seed)
    pos0 = rng0.uniform(0, L, size=(Ntotal, 2))
    angles0 = rng0.uniform(0, 2*np.pi, size=Ntotal)
    speeds0 = rng0.uniform(0, vmax, size=Ntotal)
    vel0 = np.stack([np.cos(angles0)*speeds0, np.sin(angles0)*speeds0], axis=1)
    infected_idx0 = rng0.choice(np.arange(Ntotal), size=I0, replace=False)
    target_shape_holder = [None]
    Nsteps = int(np.ceil(T / dt))

    for e in range(Nexp):
        rng = np.random.RandomState(base_seed + 1000 + e)
        pos = pos0.copy()
        vel = vel0.copy()
        states = np.zeros(Ntotal, dtype=int)  # 0 susceptible, 1 infectado, 2 recuperado
        states[infected_idx0] = 1
        SIR = np.zeros((Nsteps+1, 3), dtype=int)
        SIR[0] = [np.sum(states==0), np.sum(states==1), np.sum(states==2)]
        p_recov = 1 - np.exp(-gamma * dt)

        for step in range(1, Nsteps+1):
            pos += vel * dt
            # rebotes en los bordes (reflexión elástica)
            for dim in [0,1]:
                mask_low = pos[:,dim] < 0
                if np.any(mask_low):
                    pos[mask_low, dim] = -pos[mask_low, dim]
                    vel[mask_low, dim] *= -1
                mask_high = pos[:,dim] > L
                if np.any(mask_high):
                    pos[mask_high, dim] = 2*L - pos[mask_high, dim]
                    vel[mask_high, dim] *= -1

            infected = np.where(states==1)[0]
            susceptible = np.where(states==0)[0]
            if infected.size>0 and susceptible.size>0:
                pos_inf = pos[infected]
                pos_sus = pos[susceptible]
                # distancias al cuadrado: (len(sus), len(inf))
                d2 = np.sum((pos_sus[:, None, :] - pos_inf[None, :, :])**2, axis=2)
                within = d2 <= r*r
                k_neighbors = within.sum(axis=1)
                probs = 1 - np.exp(-beta * k_neighbors * dt)
                rand = rng.uniform(size=probs.shape)
                to_infect = susceptible[rand < probs]
                states[to_infect] = 1

            infected = np.where(states==1)[0]
            if infected.size>0:
                recover_rand = rng.uniform(size=infected.shape)
                recov_idx = infected[recover_rand < p_recov]
                states[recov_idx] = 2

            SIR[step] = [np.sum(states==0), np.sum(states==1), np.sum(states==2)]

            # frames solo para la primera réplica (para la animación de ejemplo)
            if e==0 and make_anim and (step % max(1,int(Nsteps/60))==0):
                fig, ax = plt.subplots(figsize=(5,5))
                ax.set_xlim(0, L); ax.set_ylim(0, L)
                ax.set_xticks([]); ax.set_yticks([])
                cmap = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:green'}
                for st in [0,1,2]:
                    idx = np.where(states==st)[0]
                    if idx.size>0:
                        ax.scatter(pos[idx,0], pos[idx,1], s=20, c=cmap[st], label=f'{st}', alpha=0.8)
                ax.set_title(f'Partículas: paso {step}/{Nsteps}  S={SIR[step,0]} I={SIR[step,1]} R={SIR[step,2]}')
                ax.legend(loc='upper right', markerscale=1, framealpha=0.6)
                frames_example = frames_example if frames_example is not None else []
                save_frame_consistent(frames_example, fig, target_shape_holder)
                plt.close(fig)

        all_SIR.append(SIR)

    all_SIR = np.array(all_SIR)  # (Nexp, steps+1, 3)
    mean_SIR = all_SIR.mean(axis=0)
    return mean_SIR, frames_example

# ---------------------------
# Kernel circular para CA
# ---------------------------
def circular_kernel(radius):
    r = int(np.ceil(radius))
    ys, xs = np.mgrid[-r:r+1, -r:r+1]
    mask = (xs**2 + ys**2) <= radius*radius
    mask[r, r] = 0  # excluir la propia celda
    return mask.astype(int)

# ---------------------------
# Simulación autómata celular (vectorizada con convolución)
# ---------------------------
def run_experiments_ca(Nexp, M, N, I0, T, r, beta, gamma, dt, base_seed, make_anim=True):
    rng0 = np.random.RandomState(base_seed)
    total = M*N
    flat_idx0 = rng0.choice(np.arange(total), size=I0, replace=False)
    all_SIR = []
    frames_example = None
    target_shape_holder = [None]
    kernel = circular_kernel(r)
    Nsteps = int(np.ceil(T / dt))

    for e in range(Nexp):
        rng = np.random.RandomState(base_seed + 1000 + e)
        grid = np.zeros((M,N), dtype=int)
        grid.flat[flat_idx0] = 1
        SIR = np.zeros((Nsteps+1, 3), dtype=int)
        SIR[0] = [np.sum(grid==0), np.sum(grid==1), np.sum(grid==2)]
        p_recov = 1 - np.exp(-gamma * dt)

        for step in range(1, Nsteps+1):
            infected_mask = (grid==1).astype(int)
            counts = convolve2d(infected_mask, kernel, mode='same', boundary='fill', fillvalue=0)
            sus_mask = (grid==0)
            probs = 1 - np.exp(-beta * counts * dt)
            rand = rng.uniform(size=probs.shape)
            new_grid = grid.copy()
            new_grid[(sus_mask) & (rand < probs)] = 1
            rand_rec = rng.uniform(size=grid.shape)
            new_grid[(grid==1) & (rand_rec < p_recov)] = 2
            grid = new_grid
            SIR[step] = [np.sum(grid==0), np.sum(grid==1), np.sum(grid==2)]

            if e==0 and make_anim and (step % max(1,int(Nsteps/60))==0):
                img = np.zeros((M,N,3))
                cmap = {0: (0.2,0.4,0.8), 1: (0.9,0.1,0.1), 2: (0.1,0.7,0.2)}
                for st in [0,1,2]:
                    img[grid==st] = cmap[st]
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(img, origin='lower', interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'CA: paso {step}/{Nsteps}  S={SIR[step,0]} I={SIR[step,1]} R={SIR[step,2]}')
                frames_example = frames_example if frames_example is not None else []
                save_frame_consistent(frames_example, fig, target_shape_holder)
                plt.close(fig)

        all_SIR.append(SIR)

    all_SIR = np.array(all_SIR)
    mean_SIR = all_SIR.mean(axis=0)
    return mean_SIR, frames_example

# ---------------------------
# EJEMPLO: parámetros por defecto (ajusta antes de ejecutar)
# ---------------------------
if __name__ == '__main__':
    # Parámetros (modifica según tu enunciado)
    # Partículas
    L = 1.0
    Ntotal = 150
    I0 = 4
    vmax = 0.02
    r = 0.04
    beta = 0.8
    gamma = 0.15
    dt = 1.0
    T = 100.0
    base_seed = 777
    Nexp = 4

    # Autómata celular
    M = 30; N = 30
    I0_ca = 30
    r_ca = 1.5
    beta_ca = 0.9
    gamma_ca = 0.18
    dt_ca = 1.0
    T_ca = 100.0

    out_dir = os.path.join(os.getcwd(), 'out')
    os.makedirs(out_dir, exist_ok=True)

    print("Ejecutando experimento de partículas (Nexp=", Nexp, ")...")
    mean_SIR_part, frames_part = run_experiments_particle(
        Nexp, L, Ntotal, I0, vmax, r, beta, gamma, dt, T, base_seed, make_anim=True
    )
    t_grid = np.linspace(0, T, mean_SIR_part.shape[0])
    theo_part = sir_theoretical(Ntotal - I0, I0, 0, beta, gamma, Ntotal, t_grid)

    # Guardar gráfico de comparación partículas vs ODE
    plt.figure(figsize=(8,5))
    plt.plot(t_grid, mean_SIR_part[:,0], label='S promedio (partículas)', linestyle='-')
    plt.plot(t_grid, mean_SIR_part[:,1], label='I promedio (partículas)', linestyle='-')
    plt.plot(t_grid, mean_SIR_part[:,2], label='R promedio (partículas)', linestyle='-')
    plt.plot(t_grid, theo_part[:,0], label='S teórico (ODE)', linestyle='--')
    plt.plot(t_grid, theo_part[:,1], label='I teórico (ODE)', linestyle='--')
    plt.plot(t_grid, theo_part[:,2], label='R teórico (ODE)', linestyle='--')
    plt.xlabel('Tiempo'); plt.ylabel('Individuos'); plt.legend(); plt.tight_layout()
    path_part_plot = os.path.join(out_dir, 'sir_compare_particle.png')
    plt.savefig(path_part_plot, dpi=150); plt.close()
    print("Guardado:", path_part_plot)

    # Guardar animaciones partículas (si frames existen)
    if frames_part and len(frames_part)>0:
        gif_path = os.path.join(out_dir, 'particle_sim.gif')
        imageio.mimsave(gif_path, frames_part, fps=12)
        print("GIF partículas guardado en", gif_path)
        try:
            mp4_path = os.path.join(out_dir, 'particle_sim.mp4')
            imageio.mimsave(mp4_path, frames_part, fps=12, format='ffmpeg')
            print("MP4 partículas guardado en", mp4_path)
        except Exception as e:
            print("No se pudo escribir mp4 de partículas:", e)

    print("Ejecutando experimento autómata celular (Nexp=", Nexp, ")...")
    mean_SIR_ca, frames_ca = run_experiments_ca(
        Nexp, M, N, I0_ca, T_ca, r_ca, beta_ca, gamma_ca, dt_ca, base_seed, make_anim=True
    )
    t_grid_ca = np.linspace(0, T_ca, mean_SIR_ca.shape[0])
    theo_ca = sir_theoretical(M*N - I0_ca, I0_ca, 0, beta_ca, gamma_ca, M*N, t_grid_ca)

    plt.figure(figsize=(8,5))
    plt.plot(t_grid_ca, mean_SIR_ca[:,0], label='S promedio (CA)', linestyle='-')
    plt.plot(t_grid_ca, mean_SIR_ca[:,1], label='I promedio (CA)', linestyle='-')
    plt.plot(t_grid_ca, mean_SIR_ca[:,2], label='R promedio (CA)', linestyle='-')
    plt.plot(t_grid_ca, theo_ca[:,0], label='S teórico (ODE)', linestyle='--')
    plt.plot(t_grid_ca, theo_ca[:,1], label='I teórico (ODE)', linestyle='--')
    plt.plot(t_grid_ca, theo_ca[:,2], label='R teórico (ODE)', linestyle='--')
    plt.xlabel('Tiempo'); plt.ylabel('Individuos'); plt.legend(); plt.tight_layout()
    path_ca_plot = os.path.join(out_dir, 'sir_compare_ca.png')
    plt.savefig(path_ca_plot, dpi=150); plt.close()
    print("Guardado:", path_ca_plot)

    if frames_ca and len(frames_ca)>0:
        gif_path_ca = os.path.join(out_dir, 'ca_sim.gif')
        imageio.mimsave(gif_path_ca, frames_ca, fps=12)
        print("GIF autómata celular guardado en", gif_path_ca)
        try:
            mp4_path_ca = os.path.join(out_dir, 'ca_sim.mp4')
            imageio.mimsave(mp4_path_ca, frames_ca, fps=12, format='ffmpeg')
            print("MP4 autómata celular guardado en", mp4_path_ca)
        except Exception as e:
            print("No se pudo escribir mp4 CA:", e)

    print("Listo. Archivos generados en", out_dir)
    for f in os.listdir(out_dir):
        if f.endswith(('.gif', '.mp4', '.png')):
            print("-", f)
