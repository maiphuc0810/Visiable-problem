import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# =====================
# CẤU HÌNH & SIÊU THAM SỐ
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_LOW, DOMAIN_HIGH = 0.0, 4.0
X_STAR = torch.tensor([2.5, 1.2], dtype=torch.float32, device=DEVICE)  # TRONG hình tròn 2

HIDDEN_SIZE = 128
NUM_LAYERS = 4
ACT_FUNCTION = nn.Tanh()
LR_INIT = 1e-3
EPOCHS = 5000
BATCH_SIZE = 2048
EPS_INIT, EPS_FINAL = 0.1, 1e-4

# =====================
# ĐỊNH NGHĨA HÀM VẬT CẢN (theo cách bạn đã đổi dấu)
# φ(x) < 0: TRONG vật cản
# φ(x) = 0: trên biên
# φ(x) > 0: NGOÀI vật cản
# ===================== 
def sdf_circles(xy, centers, radii):
    if centers.numel() == 0:
        return torch.full((xy.shape[0],), -1.0, device=xy.device)
    diff = xy.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.norm(dim=2)
    return (dist - radii.unsqueeze(0)).max(dim=1).values  # dist - r

# =====================
# MẠNG NƠ-RON CHO TRƯỜNG HỢP x* TRONG VẬT CẢN (φ(x*) < 0)
# =====================
class VisibilityNetInside(nn.Module):
    def __init__(self, circ_centers, circ_radii):
        super().__init__()
        self.register_buffer("circ_centers", circ_centers)
        self.register_buffer("circ_radii", circ_radii)
        self.register_buffer("x_star", X_STAR)
        
        # Tính φ(x*) - ÂM vì x* trong vật cản
        with torch.no_grad():
            phi_xs = sdf_circles(X_STAR.unsqueeze(0), circ_centers, circ_radii)[0]
            self.register_buffer("phi_xs", phi_xs)
            print(f"φ(x*) = {phi_xs:.4f} (âm - trong vật cản)")

        # Mạng neural
        layers = [nn.Linear(2, HIDDEN_SIZE), ACT_FUNCTION]
        for _ in range(NUM_LAYERS - 2):
            layers.extend([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), ACT_FUNCTION])
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)

    def _varphi(self, xy):
        return sdf_circles(xy, self.circ_centers, self.circ_radii)
    
    def forward(self, x):
        """
        Nghiệm thử: u = φ(x*) * (|φ(x)|/|φ(x*)|) cho điểm trong vật cản
        u = 0 cho điểm ngoài vật cản và trên biên
        """
        phi_x = self._varphi(x)
        
        # Khởi tạo u = 0
        u = torch.zeros_like(phi_x)
        
        # Điểm trong vật cản (φ < 0)
        inside_mask = phi_x < -1e-5
        if inside_mask.any():
            x_in = x[inside_mask]
            phi_in = phi_x[inside_mask]
            
            # Tính |φ(x)| / |φ(x*)|
            dist_factor = torch.abs(phi_in) / (torch.abs(self.phi_xs) + 1e-6)
            dist_factor = torch.clamp(dist_factor, 0.0, 1.0)
            
            # Mạng neural học phần hiệu chỉnh
            nn_out = self.net(x_in).squeeze(-1)
            
            # u = φ(x*) * dist_factor + small correction
            u_in = self.phi_xs * dist_factor + 0.01 * nn_out * dist_factor * (1 - dist_factor)
            u[inside_mask] = u_in
        
        return u

# =====================
# LẤY MẪU
# =====================
def get_samples(model, n_points):
    """Lấy mẫu tập trung vào vùng quan trọng"""
    n_inside = n_points // 2
    n_boundary = n_points // 4
    n_outside = n_points - n_inside - n_boundary
    
    all_samples = []
    
    # Mẫu trong vật cản (φ < 0)
    if n_inside > 0:
        inside_samples = []
        while len(inside_samples) < n_inside:
            cand = torch.rand(n_inside * 2, 2, device=DEVICE) * 4.0
            phi_cand = model._varphi(cand)
            inside_cand = cand[phi_cand < -0.01]
            if len(inside_cand) > 0:
                inside_samples.append(inside_cand)
        if inside_samples:
            all_samples.append(torch.cat(inside_samples)[:n_inside])
    
    # Mẫu gần biên
    if n_boundary > 0:
        for center, radius in zip(model.circ_centers, model.circ_radii):
            theta = torch.rand(n_boundary // 2, device=DEVICE) * 2 * np.pi
            x = center[0] + radius * torch.cos(theta)
            y = center[1] + radius * torch.sin(theta)
            b = torch.stack([x, y], dim=1) + torch.randn(n_boundary // 2, 2, device=DEVICE) * 0.05
            all_samples.append(b)
    
    # Mẫu ngoài vật cản (φ > 0)
    if n_outside > 0:
        outside_samples = []
        while len(outside_samples) < n_outside:
            cand = torch.rand(n_outside * 2, 2, device=DEVICE) * 4.0
            phi_cand = model._varphi(cand)
            outside_cand = cand[phi_cand > 0.01]
            if len(outside_cand) > 0:
                outside_samples.append(outside_cand)
        if outside_samples:
            all_samples.append(torch.cat(outside_samples)[:n_outside])
    
    if not all_samples:
        return torch.rand(n_points, 2, device=DEVICE)
    
    X = torch.cat(all_samples)
    if len(X) < n_points:
        extra = torch.rand(n_points - len(X), 2, device=DEVICE) * 4.0
        X = torch.cat([X, extra])
    
    return X[:n_points]

# =====================
# TÍNH LOSS CHO PDE TRONG VẬT CẢN
# =====================
def compute_loss(model, X, eps):
    X.requires_grad_(True)
    
    u = model(X)
    phi = model._varphi(X)
    
    # Phân loại điểm (theo định nghĩa φ của bạn)
    inside_mask = phi < -eps      # φ < -ε: trong vật cản
    boundary_mask = torch.abs(phi) <= eps  # |φ| ≤ ε: gần biên
    outside_mask = phi > eps       # φ > ε: ngoài vật cản
    
    losses = {}
    
    # J₁: Trong vật cản - (x-x*)·∇u = 0
    if inside_mask.any():
        X_in = X[inside_mask]
        u_in = u[inside_mask]
        
        # Tính gradient bằng sai phân hữu hạn
        h = 1e-4
        grad_u = torch.zeros(len(X_in), 2, device=DEVICE)
        
        for i, x_i in enumerate(X_in):
            x_i = x_i.detach()
            
            # Đạo hàm theo x
            x_plus = x_i + torch.tensor([h, 0.0], device=DEVICE)
            x_minus = x_i - torch.tensor([h, 0.0], device=DEVICE)
            
            with torch.no_grad():
                u_plus = model(x_plus.unsqueeze(0))
                u_minus = model(x_minus.unsqueeze(0))
            
            grad_u[i, 0] = (u_plus - u_minus) / (2 * h)
            
            # Đạo hàm theo y
            y_plus = x_i + torch.tensor([0.0, h], device=DEVICE)
            y_minus = x_i - torch.tensor([0.0, h], device=DEVICE)
            
            with torch.no_grad():
                u_plus = model(y_plus.unsqueeze(0))
                u_minus = model(y_minus.unsqueeze(0))
            
            grad_u[i, 1] = (u_plus - u_minus) / (2 * h)
        
        # (x-x*)·∇u
        dot_u = (X_in[:, 0] - model.x_star[0]) * grad_u[:, 0] + \
                (X_in[:, 1] - model.x_star[1]) * grad_u[:, 1]
        
        losses['j1'] = torch.mean(dot_u**2)
    else:
        losses['j1'] = torch.tensor(0.0, device=DEVICE)
    
    # J₂: Trên biên - u = 0
    if boundary_mask.any():
        losses['j2'] = torch.mean(u[boundary_mask]**2)
    else:
        losses['j2'] = torch.tensor(0.0, device=DEVICE)
    
    # J₃: Ngoài vật cản - u = 0
    if outside_mask.any():
        losses['j3'] = torch.mean(u[outside_mask]**2)
    else:
        losses['j3'] = torch.tensor(0.0, device=DEVICE)
    
    # Điều kiện biên tại x*
    with torch.no_grad():
        u_xs = model(model.x_star.unsqueeze(0))
    losses['j_xs'] = (u_xs - model.phi_xs)**2
    
    # Tổng loss
    losses['total'] = losses['j1'] + 10.0*losses['j2'] + 5.0*losses['j3'] + 100.0*losses['j_xs']
    
    return losses

# =====================
# VẼ KẾT QUẢ
# =====================
def plot_results(model, history):
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)
    y = torch.linspace(0, 4, grid, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Hình 1: Nghiệm u
    ax = axes[0]
    im = ax.contourf(X.cpu(), Y.cpu(), u, levels=50, cmap='viridis')
    ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='red', linewidths=2)
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='yellow', marker='*', s=200, edgecolor='black')
    ax.set_title(f'u(x) - INSIDE obstacle (φ(x*)={model.phi_xs:.3f})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # Hình 2: Vùng vật cản
    ax = axes[1]
    obstacle_mask = phi < 0  # φ < 0: trong vật cản
    im = ax.contourf(X.cpu(), Y.cpu(), obstacle_mask.astype(float), 
                     levels=[-0.5, 0.5, 1.5], colors=['lightgray', 'lightblue'], alpha=0.5)
    ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='red', linewidths=2)
    ax.contour(X.cpu(), Y.cpu(), u, levels=20, colors='black', alpha=0.3)
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='yellow', marker='*', s=200, edgecolor='black')
    ax.set_title('Obstacle region (blue) and u contours')
    ax.set_aspect('equal')
    
    # Hình 3: Loss history
    ax = axes[2]
    epochs = np.arange(len(history['total']))
    
    ax.semilogy(epochs, history['j1'], label='J₁ (inside)')
    ax.semilogy(epochs, history['j2'], label='J₂ (boundary)')
    ax.semilogy(epochs, history['j3'], label='J₃ (outside)')
    ax.semilogy(epochs, history['j_xs'], label='J_xs (BC)')
    ax.semilogy(epochs, history['total'], 'k--', label='Total', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# =====================
# HÀM CHÍNH
# =====================
def main():
    centers = torch.tensor([
        [2.5, 2.5],
        [2.5, 1.5],
    ], device=DEVICE)
    radii = torch.tensor([
        1.0,
        0.5,
    ], device=DEVICE)
    
    print("=" * 60)
    print("GIẢI BÀI TOÁN VISIBILITY - OBSERVER TRONG VẬT CẢN")
    print("=" * 60)
    print(f"Observer tại {X_STAR.cpu().numpy()}")
    
    # Kiểm tra
    with torch.no_grad():
        phi_xs = sdf_circles(X_STAR.unsqueeze(0), centers, radii)[0].item()
        print(f"φ(x*) = {phi_xs:.4f} {'✅ TRONG vật cản' if phi_xs < 0 else '❌ NGOÀI vật cản'}")
    
    model = VisibilityNetInside(centers, radii).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    history = {'j1': [], 'j2': [], 'j3': [], 'j_xs': [], 'total': []}
    best_loss = float('inf')
    
    print("\nBắt đầu training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        eps = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        
        X = get_samples(model, BATCH_SIZE)
        
        optimizer.zero_grad()
        losses = compute_loss(model, X, eps)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(losses['total'])
        
        for key in losses:
            history[key].append(losses[key].item())
        
        if losses['total'].item() < best_loss:
            best_loss = losses['total'].item()
        
        if epoch % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d} | Loss: {losses['total'].item():.2e} | "
                  f"J1: {losses['j1'].item():.2e} | J2: {losses['j2'].item():.2e} | "
                  f"J3: {losses['j3'].item():.2e} | Time: {elapsed:.1f}s")
    
    print(f"\nBest loss: {best_loss:.2e}")
    plot_results(model, history)

if __name__ == "__main__":
    main()