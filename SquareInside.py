import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# =====================
# CẤU HÌNH & SIÊU THAM SỐ
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_LOW, DOMAIN_HIGH = 0.0, 4.0
X_STAR = torch.tensor([3.2, 3.0], dtype=torch.float32, device=DEVICE)

HIDDEN_SIZE = 128
NUM_LAYERS = 4
ACT_FUNCTION = nn.GELU()
LR_INIT = 1e-3
EPOCHS = 10000
BATCH_SIZE = 2048
EPS_INIT, EPS_FINAL = 0.1, 1e-4

# =====================
# HÀM VẬT CẢN φ(x) - ĐỊNH NGHĨA PIECEWISE THEO ĐỀ BÀI
# =====================

def phi_piecewise(x, y):
    """
    Hàm vật cản φ(x1,x2) định nghĩa piecewise theo đề bài
    """
    result = torch.zeros_like(x)
    
    # Trường hợp x1 < 3: hình vuông tâm (2,3), cạnh 2 (half=1)
    mask1 = (x < 3)
    if mask1.any():
        x1, y1 = x[mask1], y[mask1]
        
        # Tính các thành phần
        dx1 = torch.abs(x1 - 2) - 1
        dy1 = torch.abs(y1 - 3) - 1
        
        # Phần min(max(...), 0)
        max_dx_dy1 = torch.max(dx1, dy1)
        min_part1 = torch.min(max_dx_dy1, torch.tensor(0.0, device=x.device))
        
        # Phần sqrt(max(...)^2 + max(...)^2)
        dx_pos1 = torch.clamp(dx1, min=0)
        dy_pos1 = torch.clamp(dy1, min=0)
        sqrt_part1 = torch.sqrt(dx_pos1**2 + dy_pos1**2)
        
        result[mask1] = min_part1 + sqrt_part1
    
    # Trường hợp x1 >= 3: hình vuông tâm (3,3), cạnh 1 (half=0.5)
    mask2 = (x >= 3)
    if mask2.any():
        x2, y2 = x[mask2], y[mask2]
        
        # Tính các thành phần
        dx2 = torch.abs(x2 - 3) - 0.5
        dy2 = torch.abs(y2 - 3) - 0.5
        
        # Phần min(max(...), 0)
        max_dx_dy2 = torch.max(dx2, dy2)
        min_part2 = torch.min(max_dx_dy2, torch.tensor(0.0, device=x.device))
        
        # Phần sqrt(max(...)^2 + max(...)^2)
        dx_pos2 = torch.clamp(dx2, min=0)
        dy_pos2 = torch.clamp(dy2, min=0)
        sqrt_part2 = torch.sqrt(dx_pos2**2 + dy_pos2**2)
        
        result[mask2] = min_part2 + sqrt_part2
    
    return result

# =====================
# ĐỊNH NGHĨA MẠNG NƠ-RON
# =====================
class VisibilityNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("x_star", X_STAR)
        
        # Tính φ(x*)
        with torch.no_grad():
            phi_at_star = phi_piecewise(
                torch.tensor([X_STAR[0]], device=DEVICE),
                torch.tensor([X_STAR[1]], device=DEVICE)
            )[0]
            self.register_buffer("varphi_xs", phi_at_star)

        # Kiến trúc mạng
        layers = [nn.Linear(2, HIDDEN_SIZE), ACT_FUNCTION]
        for _ in range(NUM_LAYERS - 2):
            layers.extend([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), ACT_FUNCTION])
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)

    def _varphi(self, xy):
        """Tính φ(x) - hàm vật cản piecewise"""
        return phi_piecewise(xy[..., 0], xy[..., 1])
    
    def _grad_varphi(self, xy):
        """Tính ∇φ(x)"""
        xy_ = xy.clone().detach().requires_grad_(True)
        phi = self._varphi(xy_)
        return torch.autograd.grad(phi.sum(), xy_, create_graph=True)[0]

    def forward(self, x):
        """ξN(x; θ) = φ(x*) + |x-x*|² * NN(x; θ)"""
        r2 = (x - self.x_star).pow(2).sum(dim=1, keepdim=True)
        return self.varphi_xs + r2.squeeze(-1) * self.net(x).squeeze(-1)

# =====================
# CÁC THÀNH PHẦN THUẬT TOÁN
# =====================

def get_importance_samples(model, n_points, mode="G1"):
    """Lấy mẫu quan trọng"""
    x_raw = torch.rand(n_points * 4, 2, device=DEVICE) * 4.0
    
    with torch.no_grad():
        u = model(x_raw)
        phi = model._varphi(x_raw)
        
        if mode == "G1": 
            weights = torch.sigmoid((u - phi) * 10) + 0.1
        elif mode == "G2": 
            weights = torch.exp(-torch.abs(u - phi) * 20) + 0.05
        else:
            weights = torch.relu(phi - u) * 50 + 0.1
    
    idx = torch.multinomial(weights.flatten() + 1e-7, n_points, replacement=True)
    return x_raw[idx]

def compute_losses(model, X1, X2, X3, eps):
    """Tính loss J₁, J₂, J₃"""
    
    # J₁: (x-x*)·∇ξN = 0 trong G₁
    X1.requires_grad_(True)
    u1, phi1 = model(X1), model._varphi(X1)
    grad_u1 = torch.autograd.grad(u1.sum(), X1, create_graph=True)[0]
    res1 = torch.sum((X1 - model.x_star) * grad_u1, dim=1)
    
    dot_phi1 = torch.sum((X1 - model.x_star) * model._grad_varphi(X1), dim=1)
    mask1 = (u1 > phi1 + 1e-5) | (dot_phi1 <= 1e-5)
    j1 = torch.mean(res1[mask1]**2) if mask1.any() else torch.tensor(0.0, device=DEVICE)

    # J₂: (x-x*)·(∇ξN-∇φ) = 0 trong G₂
    X2.requires_grad_(True)
    u2, phi2 = model(X2), model._varphi(X2)
    grad_u2 = torch.autograd.grad(u2.sum(), X2, create_graph=True)[0]
    grad_phi2 = model._grad_varphi(X2)
    
    dot_phi2 = torch.sum((X2 - model.x_star) * grad_phi2, dim=1)
    mask2 = (torch.abs(u2 - phi2) <= eps) & (dot_phi2 > 1e-5)
    res2 = torch.sum((X2 - model.x_star) * (grad_u2 - grad_phi2), dim=1)
    j2 = torch.mean(res2[mask2]**2) if mask2.any() else torch.tensor(0.0, device=DEVICE)

    # J₃: Obstacle consistency loss
    j3 = torch.mean(torch.relu(model._varphi(X3) - model(X3))**2)
    
    return j1, j2, j3

# =====================
# HÀM VẼ KẾT QUẢ
# =====================
def plot_results(model, history):
    """Vẽ kết quả"""
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)
    y = torch.linspace(0, 4, grid, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # HÌNH 1: NGHIỆM THỬ
    ax = axes[0]
    im = ax.contourf(X.cpu(), Y.cpu(), u, levels=50, cmap='viridis', alpha=0.9)
    
    # Đường ξN = 0
    if u.min() < 0 < u.max():
        u0_contour = ax.contour(X.cpu(), Y.cpu(), u, levels=[0], colors='red', linewidths=2.5)
        ax.clabel(u0_contour, inline=True, fontsize=10, fmt='$\\xi_N = 0$', colors='red')
    
    # Biên vật cản φ = 0
    ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='yellow', linewidths=2.5, linestyles='--')
    
    # Vẽ hai hình vuông (minh họa vật cản)
    rect1 = Rectangle((1, 2), 2, 2, fill=True, alpha=0.3, color='orange', edgecolor='orange', linewidth=2)
    rect2 = Rectangle((2.5, 2.5), 1, 1, fill=True, alpha=0.3, color='orange', edgecolor='orange', linewidth=2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Vẽ đường phân cách x1 = 3
    ax.axvline(x=3, color='white', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(3.05, 3.8, '$x_1 = 3$', color='white', fontsize=10, alpha=0.7)
    
    # Điểm quan sát
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='gold', marker='*', s=400, edgecolor='black', linewidth=2,
              label=f'Observer $x^*$ = ({X_STAR[0].cpu():.1f}, {X_STAR[1].cpu():.1f})', 
              zorder=10)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$\\xi_N(x; \\theta)$', fontsize=12)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('PINN Solution $\\xi_N(x;\\theta)$', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # HÌNH 2: LỊCH SỬ TRAINING
    ax2 = axes[1]
    epochs = np.arange(len(history['j1']))
    
    ax2.semilogy(epochs, history['j1'], 'b-', alpha=0.3, linewidth=1, label='$J_1$')
    ax2.semilogy(epochs, history['j2'], 'g-', alpha=0.3, linewidth=1, label='$J_2$')
    ax2.semilogy(epochs, history['j3'], 'r-', alpha=0.3, linewidth=1, label='$J_3$')
    ax2.semilogy(epochs, history['total'], 'k-', alpha=0.3, linewidth=1, label='Total')
    
    window = min(100, len(epochs)//10)
    if len(epochs) > window:
        def ma(data): 
            return np.convolve(data, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        ax2.semilogy(ma_epochs, ma(history['total']), 'k--', linewidth=3, label='Total (MA)')
    
    best_epoch = np.argmin(history['total'])
    best_loss = history['total'][best_epoch]
    ax2.scatter(best_epoch, best_loss, color='purple', s=100, marker='o', 
               label=f'Best: {best_loss:.2e}')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Training History', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# =====================
# HÀM KIỂM TRA NGHIỆM GIẢI TÍCH
# =====================
def verify_analytical_solution():
    """Kiểm tra nghiệm giải tích ψ(x) = max_{t∈[0,1]} φ(x* + t(x-x*))"""
    print("\n" + "="*70)
    print("KIỂM TRA NGHIỆM GIẢI TÍCH")
    print("="*70)
    
    # Các điểm kiểm tra
    test_points = [
        (3.2, 3.0, "Observer x*"),
        (3.8, 3.0, "Bên phải observer (cùng dòng)"),
        (3.0, 3.0, "Tâm hình vuông phải"),
        (2.0, 3.0, "Tâm hình vuông trái"),
        (3.2, 3.8, "Phía trên observer"),
        (3.2, 2.2, "Phía dưới observer"),
        (1.5, 1.5, "Góc trái dưới"),
    ]
    
    print(f"\n{'Điểm':<30} {'x1':<8} {'x2':<8} {'φ(x)':<12} {'ψ(x)':<12} {'Visible':<10}")
    print("-"*70)
    
    for x_val, y_val, name in test_points:
        x_t = torch.tensor([x_val], device=DEVICE)
        y_t = torch.tensor([y_val], device=DEVICE)
        
        phi_val = phi_piecewise(x_t, y_t).item()
        
        # Tính ψ(x) = max_{t∈[0,1]} φ(x* + t(x-x*))
        t = torch.linspace(0, 1, 500, device=DEVICE)
        x_path = X_STAR[0] + t * (x_val - X_STAR[0])
        y_path = X_STAR[1] + t * (y_val - X_STAR[1])
        phi_on_path = phi_piecewise(x_path, y_path)
        psi_val = torch.max(phi_on_path).item()
        
        visible = "Có" if psi_val <= 0 else "Không"
        
        print(f"{name:<30} {x_val:<8.2f} {y_val:<8.2f} {phi_val:<12.4f} {psi_val:<12.4f} {visible:<10}")
    
    print("\n" + "="*70)

# =====================
# MAIN
# =====================
def main():
    print("=" * 70)
    print("PINN FOR VISIBILITY PROBLEM WITH TWO SQUARE OBSTACLES")
    print("=" * 70)
    print(f"Miền Ω = [0,4] × [0,4]")
    print(f"Điểm quan sát x* = ({X_STAR[0].item():.1f}, {X_STAR[1].item():.1f})")
    print(f"Vật cản 1 (x1 < 3): Hình vuông tâm (2,3) cạnh 2")
    print(f"Vật cản 2 (x1 ≥ 3): Hình vuông tâm (3,3) cạnh 1")
    print("=" * 70)
    
    # Kiểm tra nghiệm giải tích
    verify_analytical_solution()
    
    # Khởi tạo mạng
    model = VisibilityNet2D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    
    history = {'j1': [], 'j2': [], 'j3': [], 'total': []}

    print(f"\nTraining on {DEVICE}...")
    print("-" * 50)
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        eps = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        
        X1, X2, X3 = [get_importance_samples(model, BATCH_SIZE, m) for m in ["G1", "G2", "G3"]]
        
        optimizer.zero_grad()
        j1, j2, j3 = compute_losses(model, X1, X2, X3, eps)
        loss = j1 + j2 + j3
        
        loss.backward()
        optimizer.step()

        history['j1'].append(j1.item())
        history['j2'].append(j2.item())
        history['j3'].append(j3.item())
        history['total'].append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | J1: {j1.item():.2e} | J2: {j2.item():.2e} | J3: {j3.item():.2e} | Time: {elapsed:.1f}s")

    print("-" * 50)
    print(f"Training completed in {time.time() - start_time:.1f}s")
    print(f"Best loss: {best_loss:.2e}")
    print("-" * 50)
    
    print("Plotting results...")
    plot_results(model, history)

if __name__ == "__main__":
    main()