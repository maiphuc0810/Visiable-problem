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
X_STAR = torch.tensor([0,0 ], dtype=torch.float32, device=DEVICE)  # Điểm quan sát (1.5,3.5) - nằm trong vật cản

HIDDEN_SIZE = 128  # Số node trong lớp ẩn
NUM_LAYERS = 4     # Số lớp của mạng nơ-ron
<<<<<<<< HEAD:CircleOstacle .py
ACT_FUNCTION = nn.ELU()  # Hàm kích hoạt

LR_INIT = 1e-3      # Learning rate ban đầu (lambda)
EPOCHS = 10000       # Số vòng lặp huấn luyện
BATCH_SIZE = 2048   # Số mẫu Monte Carlo mỗi batch (M)
EPS_INIT, EPS_FINAL  = 0.1, 1e-4  
# Dung sai εn - giảm dần từ 0.1 xuốngs 0.0001 (epsilon)

# =====================
# ĐỊNH NGHĨA HÀM VẬT CẢN φ(x) - (Obstacle function) 
========
ACT_FUNCTION = nn.Tanh()  # Hàm kích hoạt

LR_INIT = 1e-3      # Learning rate ban đầu (lambda)
EPOCHS = 5000       # Số vòng lặp huấn luyện
BATCH_SIZE = 2048   # Số mẫu Monte Carlo mỗi batch (M)
EPS_INIT, EPS_FINAL  = 0.1, 1e-4  
# Dung sai εn - giảm dần từ 0.1 xuống 0.0001 (epsilon)

# =====================
# ĐỊNH NGHĨA HÀM VẬT CẢN φ(x) - (Obstacle function)
>>>>>>>> 29486c2c6205bdcb7e4a24b54e6cb80c6dd95ee9:circle.py
# ===================== 
def sdf_circles(xy, centers, radii):
    """
    Hàm khoảng cách có dấu cho hình tròn
    - xy: tọa độ các điểm cần tính (N, 2)
    - centers: tâm các hình tròn
    - radii: bán kính các hình tròn
    
    Trả về: φ(x) < 0 nếu trong vật cản, =0 trên biên, >0 ngoài vật cản
    """
    if centers.numel() == 0:
        return torch.full((xy.shape[0],), -1.0, device=xy.device)
    diff = xy.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, 2)
    dist = diff.norm(dim=2)  # (N, K) - khoảng cách đến mỗi tâm
    return (radii.unsqueeze(0) - dist).max(dim=1).values  # φ(x) = max(r_i - |x - c_i|)

# =====================
# ĐỊNH NGHĨA MẠNG NƠ-RON - NGHIỆM THỬ ξN(x; θ)
# Tương ứng với dòng 3 trong Algorithm 1
# =====================
class VisibilityNet2D(nn.Module):
    def __init__(self, circ_centers, circ_radii):
        super().__init__()
        # Lưu thông tin vật cản
        self.register_buffer("circ_centers", circ_centers.to(DEVICE))
        self.register_buffer("circ_radii", circ_radii.to(DEVICE))
        self.register_buffer("x_star", X_STAR)  # Điểm quan sát x*
        
        # Tính φ(x*) - giá trị tại điểm quan sát (luôn âm vì x* nằm trong vật cản)
        with torch.no_grad():
            self.register_buffer("varphi_xs", sdf_circles(X_STAR.unsqueeze(0), circ_centers, circ_radii)[0])

        # Định nghĩa kiến trúc mạng nơ-ron: 2 đầu vào (x,y) → 1 đầu ra (giá trị vô hướng)
        layers = [nn.Linear(2, HIDDEN_SIZE), ACT_FUNCTION]
        for _ in range(NUM_LAYERS - 2):
            layers.extend([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), ACT_FUNCTION])
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)  # NN(x; θ)

    def _varphi(self, xy): 
        """Tính φ(x) - hàm vật cản"""
        return sdf_circles(xy, self.circ_centers, self.circ_radii)
    
    def _grad_varphi(self, xy):
        """Tính ∇φ(x) - gradient của hàm vật cản (dùng autograd)"""
        xy_ = xy.clone().detach().requires_grad_(True)
        phi = self._varphi(xy_)
        return torch.autograd.grad(phi.sum(), xy_, create_graph=True)[0]

    def forward(self, x):
        """
        Định nghĩa nghiệm thử ξN(x; θ) = φ(x*) + |x-x*|² * NN(x; θ)
        Công thức (3.18) - tự động thỏa mãn điều kiện ξN(x*) = φ(x*)
        """
        r2 = (x - self.x_star).pow(2).sum(dim=1, keepdim=True)  # |x-x*|²
        return self.varphi_xs + r2.squeeze(-1) * self.net(x).squeeze(-1)

# -----------------------------------------------------------------------------
# CÁC THÀNH PHẦN CỦA THUẬT TOÁN 1
# -----------------------------------------------------------------------------

def get_importance_samples(model, n_points, mode="G1"):
    """
    Lấy mẫu quan trọng (importance sampling) cho từng vùng G1, G2, G3
    Tương ứng với dòng 5,7,9 trong Algorithm 1
    """
    # Lấy mẫu thô từ phân bố đều trên [0,4] (ν1, ν2, ν3)
    x_raw = torch.rand(n_points * 4, 2, device=DEVICE) * 4.0
    
    with torch.no_grad():
        u, phi = model(x_raw), model._varphi(x_raw)  # Tính ξN và φ
        
        # Xác định trọng số cho từng vùng (xác suất được chọn)
        if mode == "G1": 
            # Vùng G1: ưu tiên điểm có u > φ hoặc gần đó
            weights = torch.sigmoid((u - phi) * 10) + 0.1
        elif mode == "G2": 
            # Vùng G2: ưu tiên điểm gần biên (|ξN-φ| nhỏ)
            weights = torch.exp(-torch.abs(u - phi) * 20) + 0.05
        else:  # G3
            # Vùng G3: ưu tiên điểm vi phạm (φ > ξN)
            weights = torch.relu(phi - u) * 50 + 0.1
    
    # Lấy mẫu theo trọng số (Monte Carlo có trọng số)
    idx = torch.multinomial(weights.flatten() + 1e-7, n_points, replacement=True)
    return x_raw[idx]

def compute_losses(model, X1, X2, X3, eps):
    """
    Tính ba thành phần loss function J₁, J₂, J₃
    Tương ứng với dòng 11 trong Algorithm 1
    """
    # ===== J₁: Eikonal loss - (x-x*)·∇ξN = 0 trong G₁ =====
    X1.requires_grad_(True)
    u1, phi1 = model(X1), model._varphi(X1)
    grad_u1 = torch.autograd.grad(u1.sum(), X1, create_graph=True)[0]  # ∇ξN
    res1 = torch.sum((X1 - model.x_star) * grad_u1, dim=1)  # (x-x*)·∇ξN
    
    # Xác định mask cho G₁: u > φ hoặc (x-x*)·∇φ ≤ 0
    dot_phi1 = torch.sum((X1 - model.x_star) * model._grad_varphi(X1), dim=1)
    mask1 = (u1 > phi1 + 1e-5) | (dot_phi1 <= 1e-5)
    
    # Tính trung bình trên Gb₁ (các mẫu trong G₁)
    j1 = torch.mean(res1[mask1]**2) if mask1.any() else torch.tensor(0.0, device=DEVICE)

    # ===== J₂: Gradient matching loss - (x-x*)·(∇ξN-∇φ) = 0 trong G₂ =====
    X2.requires_grad_(True)
    u2, phi2 = model(X2), model._varphi(X2)
    grad_u2 = torch.autograd.grad(u2.sum(), X2, create_graph=True)[0]  # ∇ξN
    grad_phi2 = model._grad_varphi(X2)  # ∇φ
    
    # Xác định mask cho G₂: |ξN-φ| ≤ εn và (x-x*)·∇φ > 0
    dot_phi2 = torch.sum((X2 - model.x_star) * grad_phi2, dim=1)
    mask2 = (torch.abs(u2 - phi2) <= eps) & (dot_phi2 > 1e-5)
    
    # Tính (x-x*)·(∇ξN-∇φ)
    res2 = torch.sum((X2 - model.x_star) * (grad_u2 - grad_phi2), dim=1)
    
    # Tính trung bình trên Gb₂
    j2 = torch.mean(res2[mask2]**2) if mask2.any() else torch.tensor(0.0, device=DEVICE)

    # ===== J₃: Obstacle consistency loss - φ - ξN (chỉ khi φ > ξN) =====
    # Tính trung bình trên Gb₃ (các điểm có φ > ξN)
    j3 = torch.mean(torch.relu(model._varphi(X3) - model(X3))**2)
    
    return j1, j2, j3

# -----------------------------------------------------------------------------
# HÀM VẼ KẾT QUẢ (Visualization)
# -----------------------------------------------------------------------------
def plot_results(model, history):
    """Vẽ nghiệm thử và lịch sử huấn luyện"""
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)
    y = torch.linspace(0, 4, grid, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()      # ξN(x)
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()  # φ(x)

    # Tạo figure với 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ===== HÌNH 1: NGHIỆM THỬ ξN(x) =====
    ax = axes[0]
    
    # Vẽ contour của ξN
    im = ax.contourf(X.cpu(), Y.cpu(), u, levels=50, cmap='viridis', alpha=0.9)
    
    # Vẽ đường ξN = 0 (màu đỏ)
    if u.min() < 0 < u.max():
        ax.contour(X.cpu(), Y.cpu(), u, levels=[0], colors='red', linewidths=2.5)
        ax.text(2, 2, 'ξ_N = 0', color='white', fontweight='bold', fontsize=12,
               bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Vẽ biên vật cản φ = 0 (màu vàng)
    if phi.max() > 0:
        ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='yellow', linewidths=2.5)

    # Đánh dấu điểm quan sát x*
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='gold', marker='*', s=400, edgecolor='white', linewidth=2,
              label=f'Observer $x^*$ = ({X_STAR[0].cpu():.1f}, {X_STAR[1].cpu():.1f})', 
              zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Trial Solution $\\xi_N(x; \\theta)$', fontsize=12, fontweight='bold')
    
    # Định dạng
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Epoch', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # ===== HÌNH 2: LỊCH SỬ TRAINING =====
    ax2 = axes[1]
    
    epochs = np.arange(len(history['j1']))
    
    # Vẽ loss components (mờ)
    ax2.semilogy(epochs, history['j1'], 'b-', alpha=0.3, linewidth=1, label='$J_1$')
    ax2.semilogy(epochs, history['j2'], 'g-', alpha=0.3, linewidth=1, label='$J_2$')
    ax2.semilogy(epochs, history['j3'], 'r-', alpha=0.3, linewidth=1, label='$J_3$')
    ax2.semilogy(epochs, history['total'], 'k-', alpha=0.3, linewidth=1, label='Total')
    
    # Vẽ đường trung bình động (moving average) - đậm hơn
    window = min(100, len(epochs)//10)
    if len(epochs) > window:
        def ma(data): return np.convolve(data, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        
        ax2.semilogy(ma_epochs, ma(history['j1']), 'b-', linewidth=2.5, label='$J_1$ (MA)')
        ax2.semilogy(ma_epochs, ma(history['j2']), 'g-', linewidth=2.5, label='$J_2$ (MA)')
        ax2.semilogy(ma_epochs, ma(history['j3']), 'r-', linewidth=2.5, label='$J_3$ (MA)')
        ax2.semilogy(ma_epochs, ma(history['total']), 'k--', linewidth=3, label='Total (MA)')
    
    # Đánh dấu epoch tốt nhất
    best_epoch = np.argmin(history['total'])
    best_loss = history['total'][best_epoch]
    ax2.scatter(best_epoch, best_loss, color='purple', s=100, marker='o', 
               label=f'Best: {best_loss:.2e}')
    ax2.axvline(x=best_epoch, color='purple', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Epoch = 10000')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training History')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# HÀM CHÍNH - THỰC THI THUẬT TOÁN
# Tương ứng với vòng lặp chính dòng 4-17 trong Algorithm 1
# -----------------------------------------------------------------------------
def main():
    # Định nghĩa vật cản: hình tròn tâm (2,2) bán kính 1
    centers =torch.tensor([
        [2.5, 2.5] ,  # Hình tròn 1: tâm (1.5, 1.5)
        [2.5, 1.5],  # Hình tròn 2: tâm (2.5, 2.5)    
    ], device=DEVICE)
    radii = torch.tensor([
        1,  # Bán kính hình 1
        0.5,  # Bán kính hình 2
    ], device=DEVICE)
    
    # Khởi tạo mạng với tham số ban đầu θ₀ (dòng 1-2)
    model = VisibilityNet2D(centers, radii).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    
    # Lưu lịch sử huấn luyện
    history = {'j1': [], 'j2': [], 'j3': [], 'total': []}

    print(f"Training on {DEVICE}...")
    print(f"Obstacle: Circle at (2, 2) with radius 1")
    print(f"Observer at {X_STAR.cpu().numpy()}")
    print(f"Domain: [0,4] × [0,4]")
    print("-" * 50)
    
    # Biến để lưu giá trị tốt nhất
    best_loss = float('inf')
    best_epoch = 0
    
    # Vòng lặp chính
    for epoch in range(1, EPOCHS + 1):
        # Cập nhật dung sai εn (giảm dần theo thời gian)
        eps = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        
        # Lấy mẫu Monte Carlo từ ba phân bố ν₁, ν₂, ν₃ (dòng 5,7,9)
        X1, X2, X3 = [get_importance_samples(model, BATCH_SIZE, m) for m in ["G1", "G2", "G3"]]
        
        # Tính loss
        optimizer.zero_grad()
        j1, j2, j3 = compute_losses(model, X1, X2, X3, eps)
        loss = j1 + j2 + j3 
        
        # Backpropagation và cập nhật θ 
        loss.backward()
        optimizer.step()

        # Lưu lịch sử
        history['j1'].append(j1.item())
        history['j2'].append(j2.item())
        history['j3'].append(j3.item())
        history['total'].append(loss.item())
        
        # Cập nhật giá trị tốt nhất
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch

        # In kết quả định kỳ
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | J1: {j1.item():.2e} | J2: {j2.item():.2e} | J3: {j3.item():.2e}")

    # Kiểm tra hội tụ (dòng 12-14) - trong code này đơn giản là chạy đủ số epoch
    print("-" * 50)
    print("Training completed!")
    print(f"Best results at epoch {best_epoch}: Loss = {best_loss:.2e}")
    print("-" * 50)
    
    # Vẽ kết quả
    print("Plotting results...")
    plot_results(model, history)

if __name__ == "__main__":
    main()