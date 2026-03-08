import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# =====================
# Config & Hyperparameters
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_LOW, DOMAIN_HIGH = 0.0, 4.0  # Miền [0,4] x [0,4]
X_STAR = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)  # Điểm quan sát (0,0)

HIDDEN_SIZE = 128
NUM_LAYERS = 4
ACT_FUNCTION = nn.ELU()

LR_INIT = 1e-3
EPOCHS =10000 
BATCH_SIZE = 2048
EPS_INIT, EPS_FINAL = 0.1, 1e-4

# -----------------------------------------------------------------------------
# Models & Physics (SDF)
# -----------------------------------------------------------------------------
def sdf_circles(xy, centers, radii):
    if centers.numel() == 0:
        return torch.full((xy.shape[0],), -1.0, device=xy.device)
    diff = xy.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.norm(dim=2)
    return (radii.unsqueeze(0) - dist).max(dim=1).values

class VisibilityNet2D(nn.Module):
    def __init__(self, circ_centers, circ_radii):
        super().__init__()
        self.register_buffer("circ_centers", circ_centers.to(DEVICE))
        self.register_buffer("circ_radii", circ_radii.to(DEVICE))
        self.register_buffer("x_star", X_STAR)
        
        with torch.no_grad():
            self.register_buffer("varphi_xs", sdf_circles(X_STAR.unsqueeze(0), circ_centers, circ_radii)[0])

        layers = [nn.Linear(2, HIDDEN_SIZE), ACT_FUNCTION]
        for _ in range(NUM_LAYERS - 2):
            layers.extend([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), ACT_FUNCTION])
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)

    def _varphi(self, xy): return sdf_circles(xy, self.circ_centers, self.circ_radii)
    
    def _grad_varphi(self, xy):
        xy_ = xy.clone().detach().requires_grad_(True)
        phi = self._varphi(xy_)
        return torch.autograd.grad(phi.sum(), xy_, create_graph=True)[0]

    def forward(self, x):
        r2 = (x - self.x_star).pow(2).sum(dim=1, keepdim=True)
        return self.varphi_xs + r2.squeeze(-1) * self.net(x).squeeze(-1)

# -----------------------------------------------------------------------------
# Algorithm 1 Components
# -----------------------------------------------------------------------------
def get_importance_samples(model, n_points, mode="G1"):
    x_raw = torch.rand(n_points * 4, 2, device=DEVICE) * 4.0  # Scale lên [0,4]
    with torch.no_grad():
        u, phi = model(x_raw), model._varphi(x_raw)
        if mode == "G1": weights = torch.sigmoid((u - phi) * 10) + 0.1
        elif mode == "G2": weights = torch.exp(-torch.abs(u - phi) * 20) + 0.05
        else: weights = torch.relu(phi - u) * 50 + 0.1
    idx = torch.multinomial(weights.flatten() + 1e-7, n_points, replacement=True)
    return x_raw[idx]

def compute_losses(model, X1, X2, X3, eps):
    X1.requires_grad_(True)
    u1, phi1 = model(X1), model._varphi(X1)
    grad_u1 = torch.autograd.grad(u1.sum(), X1, create_graph=True)[0]
    res1 = torch.sum((X1 - model.x_star) * grad_u1, dim=1)
    dot_phi1 = torch.sum((X1 - model.x_star) * model._grad_varphi(X1), dim=1)
    mask1 = (u1 > phi1 + 1e-5) | (dot_phi1 <= 1e-5)
    j1 = torch.mean(res1[mask1]**2) if mask1.any() else torch.tensor(0.0, device=DEVICE)

    X2.requires_grad_(True)
    u2, phi2 = model(X2), model._varphi(X2)
    grad_u2 = torch.autograd.grad(u2.sum(), X2, create_graph=True)[0]
    grad_phi2 = model._grad_varphi(X2)
    dot_phi2 = torch.sum((X2 - model.x_star) * grad_phi2, dim=1)
    mask2 = (torch.abs(u2 - phi2) <= eps) & (dot_phi2 > 1e-5)
    res2 = torch.sum((X2 - model.x_star) * (grad_u2 - grad_phi2), dim=1)
    j2 = torch.mean(res2[mask2]**2) if mask2.any() else torch.tensor(0.0, device=DEVICE)

    j3 = torch.mean(torch.relu(model._varphi(X3) - model(X3))**2)
    return j1, j2, j3

# -----------------------------------------------------------------------------
# Visualization - Improved version
# -----------------------------------------------------------------------------
def plot_results(model, history):
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)  # Miền [0,4]
    y = torch.linspace(0, 4, grid, device=DEVICE)  # Miền [0,4]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()

    # Tạo figure với 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ===== HÌNH 1: NGHIỆM THỬ VỚI MÀU SẮC GIỐNG VIRIDIS =====
    ax = axes[0]
    
    # Dùng colormap 'viridis' 
    im = ax.contourf(X.cpu(), Y.cpu(), u, levels=50, cmap='viridis', alpha=0.9, extend='both')
    
    # Vẽ đường zero level set (u = 0) - KHÔNG dùng clabel
    if u.min() < 0 < u.max():
        zero_contour = ax.contour(X.cpu(), Y.cpu(), u, levels=[0], 
                                  colors='red', linewidths=2.5, linestyles='-')
        # Thêm text bằng tay thay vì dùng clabel
        ax.text(2, 2, 'ξ_N = 0', color='white', fontweight='bold', fontsize=12,
               bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Vẽ vật cản (phi > 0)
    obstacle_mask = phi > 0
    if obstacle_mask.any():
        # Vẽ biên vật cản (phi = 0) - KHÔNG dùng clabel
        boundary = ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], 
                             colors='yellow', linewidths=2.5, linestyles='-')

    # Đánh dấu điểm quan sát x*
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='gold', marker='*', s=400, edgecolor='white', linewidth=2,
              label=f'Observer $x^*$ = ({X_STAR[0].cpu():.1f}, {X_STAR[1].cpu():.1f})', 
              zorder=10, alpha=0.95)
    
    # Thêm colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Trial Solution $\\xi_N(x; \\theta)$', fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Epoch n = 10000', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # ===== HÌNH 2: LỊCH SỬ TRAINING =====
    ax2 = axes[1]
    
    epochs = np.arange(len(history['j1']))
    
    # Vẽ loss components
    ax2.semilogy(epochs, history['j1'], 'b-', alpha=0.3, linewidth=1, label='$J_1$')
    ax2.semilogy(epochs, history['j2'], 'g-', alpha=0.3, linewidth=1, label='$J_2$')
    ax2.semilogy(epochs, history['j3'], 'r-', alpha=0.3, linewidth=1, label='$J_3$')
    ax2.semilogy(epochs, history['total'], 'k-', alpha=0.3, linewidth=1, label='Total')
    
    # Vẽ đường trung bình
    window = min(100, len(epochs)//10)
    if len(epochs) > window:
        def ma(data): return np.convolve(data, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        
        ax2.semilogy(ma_epochs, ma(history['j1']), 'b-', linewidth=2.5, label='$J_1$ (MA)')
        ax2.semilogy(ma_epochs, ma(history['j2']), 'g-', linewidth=2.5, label='$J_2$ (MA)')
        ax2.semilogy(ma_epochs, ma(history['j3']), 'r-', linewidth=2.5, label='$J_3$ (MA)')
        ax2.semilogy(ma_epochs, ma(history['total']), 'k--', linewidth=3, label='Total (MA)')
    
    # Đánh dấu best epoch
    best_epoch = np.argmin(history['total'])
    best_loss = history['total'][best_epoch]
    ax2.scatter(best_epoch, best_loss, color='purple', s=100, marker='o', 
               label=f'Best: {best_loss:.2e}')
    ax2.axvline(x=best_epoch, color='purple', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training History')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    # Vật cản hình tròn tâm (2,2) bán kính 1
    centers = torch.tensor([[2.0, 2.0]], device=DEVICE)
    radii = torch.tensor([1.0], device=DEVICE)
    
    model = VisibilityNet2D(centers, radii).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    
    history = {'j1': [], 'j2': [], 'j3': [], 'total': []}

    print(f"Training on {DEVICE}...")
    print(f"Obstacle: Circle at (2, 2) with radius 1")
    print(f"Observer at {X_STAR.cpu().numpy()}")
    print(f"Domain: [0,4] × [0,4]")
    print("-" * 50)
    
    # Biến để lưu giá trị tốt nhất
    best_loss = float('inf')
    best_j1 = float('inf')
    best_j2 = float('inf')
    best_j3 = float('inf')
    best_epoch = 0
    
    for epoch in range(1, EPOCHS + 1):
        eps = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        X1, X2, X3 = [get_importance_samples(model, BATCH_SIZE, m) for m in ["G1", "G2", "G3"]]
        
        optimizer.zero_grad()
        j1, j2, j3 = compute_losses(model, X1, X2, X3, eps)
        loss = j1 + j2 + 2.0 * j3
        loss.backward()
        optimizer.step()

        # Lưu history
        j1_val = j1.item()
        j2_val = j2.item()
        j3_val = j3.item()
        loss_val = loss.item()
        
        history['j1'].append(j1_val)
        history['j2'].append(j2_val)
        history['j3'].append(j3_val)
        history['total'].append(loss_val)
        
        # Cập nhật giá trị tốt nhất
        if loss_val < best_loss:
            best_loss = loss_val
            best_j1 = j1_val
            best_j2 = j2_val
            best_j3 = j3_val
            best_epoch = epoch

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss_val:.2e} | J1: {j1_val:.2e} | J2: {j2_val:.2e} | J3: {j3_val:.2e}")

    print("-" * 50)
    print("Training completed!")
    print(f"Best results at epoch {best_epoch}:")
    print(f"  Minimum Loss: {best_loss:.2e}")
    print(f"  Corresponding J1: {best_j1:.2e}")
    print(f"  Corresponding J2: {best_j2:.2e}")
    print(f"  Corresponding J3: {best_j3:.2e}")
    print("-" * 50)
    print("Plotting results...")
    plot_results(model, history)

if __name__ == "__main__":
    main()