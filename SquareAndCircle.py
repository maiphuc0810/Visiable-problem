import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# =====================
# CẤU HÌNH & SIÊU THAM SỐ
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_LOW, DOMAIN_HIGH = 0.0, 4.0  # Miền tính toán [0,4] x [0,4]
X_STAR = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)  # Điểm quan sát (0,0)

HIDDEN_SIZE = 128  # Số node trong lớp ẩn
NUM_LAYERS = 4     # Số lớp của mạng nơ-ron
ACT_FUNCTION =nn.Sigmoid()  # Hàm kích hoạt

LR_INIT = 1e-3      # Learning rate ban đầu (lambda)
EPOCHS = 10000       # Số vòng lặp huấn luyện
BATCH_SIZE = 2048   # Số mẫu Monte Carlo mỗi batch (M)
EPS_INIT, EPS_FINAL = 0.1, 1e-4  
# Dung sai εn - giảm dần từ 0.1 xuống 0.0001 (epsilon)

# =====================
# ĐỊNH NGHĨA HÀM VẬT CẢN TỔNG HỢP φ(x) - (Combined Obstacle function)
# ===================== 

def sdf_square(xy, center, half_side):
    """
    Hàm khoảng cách có dấu (Signed Distance Function - SDF) cho hình vuông
    """
    cx, cy = center
    x, y = xy[:, 0], xy[:, 1]
    
    dx = torch.abs(x - cx) - half_side
    dy = torch.abs(y - cy) - half_side
    
    inside = (dx <= 0) & (dy <= 0)
    outside = ~inside
    
    phi_outside = -torch.sqrt(dx.clamp(min=0)**2 + dy.clamp(min=0)**2)
    phi_inside = -torch.max(dx, dy)
    
    phi = torch.where(inside, phi_inside, phi_outside)
    return phi

def sdf_circle(xy, center, radius):
    """
    Hàm khoảng cách có dấu cho hình tròn
    """
    diff = xy - center
    dist = diff.norm(dim=1)
    return radius - dist  # φ(x) = r - |x - c|

def combined_obstacle(xy, circle_centers, circle_radii, square_center, square_half_side):
    """
    Hàm vật cản tổng hợp kết hợp hình tròn và hình vuông
    φ(x) = max(φ_circle1, φ_circle2, φ_square)
    
    Args:
        xy: tensor (N, 2) tọa độ các điểm
        circle_centers: tensor (2, 2) tâm các hình tròn
        circle_radii: tensor (2,) bán kính các hình tròn
        square_center: tuple (cx, cy) tâm hình vuông
        square_half_side: float nửa cạnh hình vuông
    
    Returns:
        tensor (N,) giá trị φ(x) (dương trong vật cản)
    """
    # Tính φ cho từng vật cản
    phi_circle1 = sdf_circle(xy, circle_centers[0], circle_radii[0])
    phi_circle2 = sdf_circle(xy, circle_centers[1], circle_radii[1])
    phi_square = sdf_square(xy, square_center, square_half_side)
    
    # Lấy giá trị lớn nhất (vì vật cản là hợp của các miền)
    phi = torch.maximum(torch.maximum(phi_circle1, phi_circle2), phi_square)
    
    return phi

# =====================
# ĐỊNH NGHĨA MẠNG NƠ-RON - NGHIỆM THỬ ξN(x; θ)
# =====================
class VisibilityNet2D(nn.Module):
    def __init__(self, circle_centers, circle_radii, square_center, square_half_side):
        super().__init__()
        # Lưu thông tin vật cản
        self.register_buffer("circle_centers", circle_centers.to(DEVICE))
        self.register_buffer("circle_radii", circle_radii.to(DEVICE))
        self.square_center = square_center
        self.square_half_side = square_half_side
        self.register_buffer("x_star", X_STAR)  # Điểm quan sát x*
        
        # Tính φ(x*) - giá trị tại điểm quan sát
        with torch.no_grad():
            phi_xs = combined_obstacle(
                X_STAR.unsqueeze(0), 
                circle_centers, circle_radii, 
                torch.tensor(square_center, device=DEVICE), 
                torch.tensor(square_half_side, device=DEVICE)
            )[0]
            self.register_buffer("varphi_xs", phi_xs)

        # Định nghĩa kiến trúc mạng nơ-ron
        layers = [nn.Linear(2, HIDDEN_SIZE), ACT_FUNCTION]
        for _ in range(NUM_LAYERS - 2):
            layers.extend([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), ACT_FUNCTION])
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)

    def _varphi(self, xy): 
        """Tính φ(x) - hàm vật cản tổng hợp"""
        return combined_obstacle(
            xy, 
            self.circle_centers, self.circle_radii,
            self.square_center, self.square_half_side
        )
    
    def _grad_varphi(self, xy):
        """Tính ∇φ(x) - gradient của hàm vật cản (dùng autograd)"""
        xy_ = xy.clone().detach().requires_grad_(True)
        phi = self._varphi(xy_)
        return torch.autograd.grad(phi.sum(), xy_, create_graph=True)[0]

    def forward(self, x):
        """
        Định nghĩa nghiệm thử ξN(x; θ) = φ(x*) + |x-x*|² * NN(x; θ)
        """
        r2 = (x - self.x_star).pow(2).sum(dim=1, keepdim=True)  # |x-x*|²
        return self.varphi_xs + r2.squeeze(-1) * self.net(x).squeeze(-1)

# -----------------------------------------------------------------------------
# CÁC THÀNH PHẦN CỦA THUẬT TOÁN
# -----------------------------------------------------------------------------

def get_importance_samples(model, n_points, mode="G1"):
    """
    Lấy mẫu quan trọng (importance sampling) cho từng vùng G1, G2, G3
    """
    # Lấy mẫu thô từ phân bố đều trên [0,4] (ν1, ν2, ν3)
    x_raw = torch.rand(n_points * 4, 2, device=DEVICE) * 4.0
    
    with torch.no_grad():
        u, phi = model(x_raw), model._varphi(x_raw)  # Tính ξN và φ
        
        # Xác định trọng số cho từng vùng
        if mode == "G1": 
            weights = torch.sigmoid((u - phi) * 10) + 0.1
        elif mode == "G2": 
            weights = torch.exp(-torch.abs(u - phi) * 20) + 0.05
        else:  # G3
            weights = torch.relu(phi - u) * 50 + 0.1
    
    # Lấy mẫu theo trọng số
    idx = torch.multinomial(weights.flatten() + 1e-7, n_points, replacement=True)
    return x_raw[idx]

def compute_losses(model, X1, X2, X3, eps):
    """
    Tính ba thành phần loss function J₁, J₂, J₃
    """
    # ===== J₁: (x-x*)·∇ξN = 0 trong G₁ =====
    X1.requires_grad_(True)
    u1, phi1 = model(X1), model._varphi(X1)
    grad_u1 = torch.autograd.grad(u1.sum(), X1, create_graph=True)[0]
    res1 = torch.sum((X1 - model.x_star) * grad_u1, dim=1)
    
    dot_phi1 = torch.sum((X1 - model.x_star) * model._grad_varphi(X1), dim=1)
    mask1 = (u1 > phi1 + 1e-5) | (dot_phi1 <= 1e-5)
    
    j1 = torch.mean(res1[mask1]**2) if mask1.any() else torch.tensor(0.0, device=DEVICE)

    # ===== J₂: (x-x*)·(∇ξN-∇φ) = 0 trong G₂ =====
    X2.requires_grad_(True)
    u2, phi2 = model(X2), model._varphi(X2)
    grad_u2 = torch.autograd.grad(u2.sum(), X2, create_graph=True)[0]
    grad_phi2 = model._grad_varphi(X2)
    
    dot_phi2 = torch.sum((X2 - model.x_star) * grad_phi2, dim=1)
    mask2 = (torch.abs(u2 - phi2) <= eps) & (dot_phi2 > 1e-5)
    
    res2 = torch.sum((X2 - model.x_star) * (grad_u2 - grad_phi2), dim=1)
    j2 = torch.mean(res2[mask2]**2) if mask2.any() else torch.tensor(0.0, device=DEVICE)

    # ===== J₃: Obstacle consistency loss - φ - ξN (chỉ khi φ > ξN) =====
    j3 = torch.mean(torch.relu(model._varphi(X3) - model(X3))**2)
    
    return j1, j2, j3

# -----------------------------------------------------------------------------
# HÀM VẼ KẾT QUẢ
# -----------------------------------------------------------------------------
def plot_results(model, history):
    """Vẽ nghiệm thử và lịch sử huấn luyện"""
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)
    y = torch.linspace(0, 4, grid, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ===== HÌNH 1: NGHIỆM THỬ ξN(x) =====
    ax = axes[0]
    
    # Vẽ contour của ξN
    im = ax.contourf(X.cpu(), Y.cpu(), u, levels=50, cmap='viridis', alpha=0.9)
    
    # Vẽ đường ξN = 0
    if u.min() < 0 < u.max():
        cs = ax.contour(X.cpu(), Y.cpu(), u, levels=[0], colors='red', linewidths=2.5)
        ax.clabel(cs, inline=True, fontsize=10, fmt='ξ_N = 0')
    
    # Vẽ biên vật cản φ = 0
    if phi.max() > 0:
        cs_phi = ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='yellow', linewidths=2.5)
        ax.clabel(cs_phi, inline=True, fontsize=10, fmt='φ = 0')

    # Đánh dấu điểm quan sát x*
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='gold', marker='*', s=400, edgecolor='white', linewidth=2,
              label=f'Observer $x^*$ = ({X_STAR[0].cpu():.1f}, {X_STAR[1].cpu():.1f})', 
              zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Trial Solution $\\xi_N(x; \\theta)$', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'Combined Obstacles: 2 Circles + Square (Epoch {EPOCHS})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # ===== HÌNH 2: LỊCH SỬ TRAINING =====
    ax2 = axes[1]
    
    epochs = np.arange(len(history['j1']))
    
    # Vẽ loss components
    ax2.semilogy(epochs, history['j1'], 'b-', alpha=0.5, linewidth=1.5, label='$J_1$')
    ax2.semilogy(epochs, history['j2'], 'g-', alpha=0.5, linewidth=1.5, label='$J_2$')
    ax2.semilogy(epochs, history['j3'], 'r-', alpha=0.5, linewidth=1.5, label='$J_3$')
    ax2.semilogy(epochs, history['total'], 'k-', linewidth=2, label='Total')
    
    # Đánh dấu epoch tốt nhất
    best_epoch = np.argmin(history['total'])
    best_loss = history['total'][best_epoch]
    ax2.scatter(best_epoch, best_loss, color='purple', s=100, marker='o', 
               label=f'Best: {best_loss:.2e} @ epoch {best_epoch}')
    ax2.axvline(x=best_epoch, color='purple', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training History - Combined Obstacles')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('PINN Solution for Visibility Problem with Multiple Obstacles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('combined_obstacles_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# HÀM CHÍNH
# -----------------------------------------------------------------------------
def main():
    # Định nghĩa vật cản:
    # 1. Hai hình tròn
    circle_centers = torch.tensor([
        [2, 2],  # Hình tròn 1: tâm (2.5, 2.5)
        [2.6, 1.2],  # Hình tròn 2: tâm (1.5, 1.5)
    ], device=DEVICE)
    circle_radii = torch.tensor([
        1,  # Bán kính hình tròn 1
        0.5,  # Bán kính hình tròn 2
    ], device=DEVICE)
    
    # 2. Một hình vuông
    square_center = (0.8, 3.0)  # Tâm hình vuông tại (2.0, 3.0)
    square_half_side = 0.3       # Nửa cạnh = 0.5 (cạnh = 1.0)
    
    # Khởi tạo mạng
    model = VisibilityNet2D(circle_centers, circle_radii, square_center, square_half_side).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)
    
    # Lưu lịch sử huấn luyện
    history = {'j1': [], 'j2': [], 'j3': [], 'total': []}

    print(f"Training on {DEVICE}...")
    print("Obstacles configuration:")
    print(f"  - Circle 1: center ({circle_centers[0,0].item():.1f}, {circle_centers[0,1].item():.1f}), radius {circle_radii[0].item():.1f}")
    print(f"  - Circle 2: center ({circle_centers[1,0].item():.1f}, {circle_centers[1,1].item():.1f}), radius {circle_radii[1].item():.1f}")
    print(f"  - Square: center ({square_center[0]:.1f}, {square_center[1]:.1f}), half-side {square_half_side:.1f}")
    print(f"Observer at {X_STAR.cpu().numpy()}")
    print(f"Domain: [0,4] × [0,4]")
    print("-" * 60)
    
    best_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # Cập nhật dung sai εn
        eps = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        
        # Lấy mẫu Monte Carlo
        X1, X2, X3 = [get_importance_samples(model, BATCH_SIZE, m) for m in ["G1", "G2", "G3"]]
        
        # Tính loss
        optimizer.zero_grad()
        j1, j2, j3 = compute_losses(model, X1, X2, X3, eps)
        loss = j1 + j2 + 2.0 * j3  # Trọng số cao hơn cho J3
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        # Lưu lịch sử
        history['j1'].append(j1.item())
        history['j2'].append(j2.item())
        history['j3'].append(j3.item())
        history['total'].append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model_combined.pth')

        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | J1: {j1.item():.2e} | J2: {j2.item():.2e} | J3: {j3.item():.2e} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print("-" * 60)
    print("Training completed!")
    print(f"Best results at epoch {best_epoch}: Loss = {best_loss:.2e}")
    print(f"Total training time: {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
    print("-" * 60)
    
    # Load model tốt nhất và vẽ kết quả
    model.load_state_dict(torch.load('best_model_combined.pth'))
    print("Plotting results...")
    plot_results(model, history)
    
    # In thông tin về các vùng
    with torch.no_grad():
        test_points = torch.rand(10000, 2, device=DEVICE) * 4.0
        u_test = model(test_points)
        phi_test = model._varphi(test_points)
        
        print("\nStatistics:")
        print(f"  - % points with u > φ: {(u_test > phi_test).float().mean().item()*100:.1f}%")
        print(f"  - % points with u ≈ φ (|u-φ| < 0.01): {(torch.abs(u_test - phi_test) < 0.01).float().mean().item()*100:.1f}%")
        print(f"  - % points violating φ > u: {(phi_test > u_test + 0.01).float().mean().item()*100:.1f}%")

if __name__ == "__main__":
    main()