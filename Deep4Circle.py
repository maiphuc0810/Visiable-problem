import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===================================================================
# 1. THÔNG SỐ BÀI TOÁN
# ===================================================================
X_STAR = np.array([[1.5, 1.5]], dtype=np.float32)  # Điểm quan sát mới

# Bốn hình tròn vật cản
CENTERS = np.array([
    [3.0, 3.0],   # Hình tròn 1
    [3.0, 1.8],   # Hình tròn 2
    [0.8, 0.5],   # Hình tròn 3
    [1.0, 3.0]    # Hình tròn 4
], dtype=np.float32)
RADII = np.array([1.0, 0.5, 0.3, 0.6], dtype=np.float32)  # Bán kính

# Tính φ(x*) = min_i (R_i - |x* - c_i|)
distances = []
for i in range(len(CENTERS)):
    dist = np.sqrt((X_STAR[0,0] - CENTERS[i,0])**2 + (X_STAR[0,1] - CENTERS[i,1])**2)
    distances.append(dist)
    phi_i = RADII[i] - dist
    print(f"Circle {i+1}: center {CENTERS[i]}, radius {RADII[i]}, distance = {dist:.4f}, phi = {phi_i:.4f}")

PHI_X_STAR = max([RADII[i] - distances[i] for i in range(len(RADII))])  # Lấy max (gần biên nhất)
print(f"\nφ(x*) = {PHI_X_STAR:.4f}")

EPS_INIT = 0.1
EPS_FINAL = 1e-4
epsilon_v = torch.tensor([EPS_INIT], dtype=torch.float32, requires_grad=False)

# ===================================================================
# 2. ĐỊNH NGHĨA HÌNH HỌC (mở rộng để chứa 4 hình tròn)
# ===================================================================
geom = dde.geometry.Rectangle([0, 0], [4, 4])

# ===================================================================
# 3. HÀM TÍNH φ(x) VÀ GRADIENT CHO NHIỀU HÌNH TRÒN
# ===================================================================
def compute_phi_grad(x):
    """
    Tính φ(x) = max_i (R_i - |x - c_i|)
    và gradient của nó (dùng softmax để xấp xỉ)
    """
    device = x.device
    centers_t = torch.tensor(CENTERS, dtype=torch.float32, device=device)
    radii_t = torch.tensor(RADII, dtype=torch.float32, device=device)
    
    # Tính khoảng cách đến từng tâm
    diff = x.unsqueeze(1) - centers_t.unsqueeze(0)  # (N, K, 2)
    dist = torch.norm(diff, dim=2)  # (N, K)
    
    # Tính SDF cho từng hình tròn: R_i - |x - c_i|
    sdf = radii_t.unsqueeze(0) - dist  # (N, K)
    
    # φ(x) = max_k sdf(x)
    phi, indices = torch.max(sdf, dim=1, keepdim=True)  # (N, 1)
    
    # Tính gradient bằng cách lấy gradient của hình tròn có giá trị max
    # (dùng softmax để gradient mượt hơn)
    temperature = 10.0
    softmax_weights = torch.softmax(sdf * temperature, dim=1)  # (N, K)
    
    # Gradient cho mỗi hình tròn
    grad_phi = torch.zeros_like(x)
    
    for k in range(len(RADII)):
        # Gradient của (R_k - |x - c_k|) là -(x - c_k)/|x - c_k|
        center_k = centers_t[k]
        dist_k = dist[:, k:k+1]
        mask_k = (dist_k > 0).float()
        grad_k = -(x - center_k) / (dist_k + 1e-12) * mask_k
        
        # Đóng góp theo trọng số softmax
        grad_phi = grad_phi + softmax_weights[:, k:k+1] * grad_k
    
    return phi, grad_phi

# ===================================================================
# 4. HÀM LẤY MẪU THEO IMPORTANCE SAMPLING
# ===================================================================
class ImportanceSamplingData:
    def __init__(self, geometry):
        self.geom = geometry
        self.model = None
        
    def set_model(self, model):
        self.model = model
        
    def train_next_batch(self, batch_size=2048):
        # Lấy mẫu raw points
        X1 = self.geom.random_points(4 * batch_size)
        X2 = self.geom.random_points(4 * batch_size)
        X3 = self.geom.random_points(4 * batch_size)
        
        if self.model is not None:
            # Tính weights với model hiện tại
            with torch.no_grad():
                # Chuyển sang tensor
                device = next(self.model.parameters()).device
                
                X1_t = torch.from_numpy(X1).float().to(device)
                X2_t = torch.from_numpy(X2).float().to(device)
                X3_t = torch.from_numpy(X3).float().to(device)
                
                # Forward pass
                u1 = self.model(X1_t)
                u2 = self.model(X2_t)
                u3 = self.model(X3_t)
                
                # Tính φ
                phi1, _ = compute_phi_grad(X1_t)
                phi2, _ = compute_phi_grad(X2_t)
                phi3, _ = compute_phi_grad(X3_t)
                
                # Tính weights
                weights1 = torch.sigmoid((u1 - phi1) * 10) + 0.1
                weights2 = torch.exp(-torch.abs(u2 - phi2) * 20) + 0.05
                weights3 = torch.relu(phi3 - u3) * 50 + 0.1
                
                # Lấy mẫu theo weights
                weights1 = (weights1.flatten() + 1e-7).cpu().numpy()
                weights1 = weights1 / weights1.sum()
                weights2 = (weights2.flatten() + 1e-7).cpu().numpy()
                weights2 = weights2 / weights2.sum()
                weights3 = (weights3.flatten() + 1e-7).cpu().numpy()
                weights3 = weights3 / weights3.sum()
                
                idx1 = np.random.choice(len(weights1), batch_size, p=weights1)
                idx2 = np.random.choice(len(weights2), batch_size, p=weights2)
                idx3 = np.random.choice(len(weights3), batch_size, p=weights3)
                
                return X1[idx1], X2[idx2], X3[idx3]
        
        # Fallback: random uniform
        return (self.geom.random_points(batch_size),
                self.geom.random_points(batch_size),
                self.geom.random_points(batch_size))

# ===================================================================
# 5. TẠO MODEL DEEPXDE (không cần data)
# ===================================================================
net = dde.nn.FNN([2] + [128] * 4 + [1], "tanh", "Glorot uniform")

def output_transform(x, y):
    device = x.device
    x_star_t = torch.tensor(X_STAR, dtype=torch.float32, device=device)
    r2 = torch.sum(torch.square(x - x_star_t), dim=1, keepdim=True)
    return PHI_X_STAR + r2 * y

net.apply_output_transform(output_transform)

# Tạo model mà không cần data
model = dde.Model(None, net)

# Tạo importance sampling data handler
sampler = ImportanceSamplingData(geom)
sampler.set_model(net)  # Set model là net

# ===================================================================
# 6. CALLBACK CHO EPSILON
# ===================================================================
class EpsilonCallback:
    def on_epoch_begin(self, epoch):
        new_val = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        epsilon_v.fill_(new_val)
        return epsilon_v.item()

# ===================================================================
# 7. HUẤN LUYỆN THỦ CÔNG
# ===================================================================
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epsilon_cb = EpsilonCallback()

history = {'j1': [], 'j2': [], 'j3': [], 'total': []}

print("\n" + "="*60)
print("Training configuration:")
print("="*60)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Four circular obstacles:")
for i in range(len(CENTERS)):
    print(f"  - Circle {i+1}: center {CENTERS[i]}, radius {RADII[i]}")
print(f"Observer at {X_STAR[0]}")
print(f"φ(x*) = {PHI_X_STAR:.4f}")
print("="*60)
print("\nStarting training...")
print("-" * 60)

for epoch in range(1, 10001):  # Tăng số epoch để đảm bảo hội tụ
    # Update epsilon
    eps = epsilon_cb.on_epoch_begin(epoch)
    
    # Get importance samples
    X1, X2, X3 = sampler.train_next_batch(2048)
    
    # Convert to tensors
    device = next(net.parameters()).device
    X1_t = torch.from_numpy(X1).float().to(device).requires_grad_(True)
    X2_t = torch.from_numpy(X2).float().to(device).requires_grad_(True)
    X3_t = torch.from_numpy(X3).float().to(device).requires_grad_(True)
    
    # Forward pass
    y1 = net(X1_t)
    y2 = net(X2_t)
    y3 = net(X3_t)
    
    # ========== Tính J1 ==========
    du_dx1 = torch.autograd.grad(y1.sum(), X1_t, create_graph=True)[0]
    du_dx11, du_dx21 = du_dx1[:, 0:1], du_dx1[:, 1:2]
    
    x_star_t = torch.tensor(X_STAR, dtype=torch.float32, device=device)
    dx1 = X1_t[:, 0:1] - x_star_t[:, 0:1]
    dy1 = X1_t[:, 1:2] - x_star_t[:, 1:2]
    
    dot_grad_u1 = dx1 * du_dx11 + dy1 * du_dx21
    
    phi1, grad_phi1 = compute_phi_grad(X1_t)
    dot_grad_phi1 = dx1 * grad_phi1[:, 0:1] + dy1 * grad_phi1[:, 1:2]
    
    mask1 = (y1 > phi1) | (dot_grad_phi1 <= 1e-5)
    j1 = torch.mean(dot_grad_u1[mask1]**2) if mask1.any() else torch.tensor(0.0, device=device)
    
    # ========== Tính J2 ==========
    du_dx2 = torch.autograd.grad(y2.sum(), X2_t, create_graph=True)[0]
    du_dx12, du_dx22 = du_dx2[:, 0:1], du_dx2[:, 1:2]
    
    dx2 = X2_t[:, 0:1] - x_star_t[:, 0:1]
    dy2 = X2_t[:, 1:2] - x_star_t[:, 1:2]
    
    dot_grad_u2 = dx2 * du_dx12 + dy2 * du_dx22
    
    phi2, grad_phi2 = compute_phi_grad(X2_t)
    dot_grad_phi2 = dx2 * grad_phi2[:, 0:1] + dy2 * grad_phi2[:, 1:2]
    
    mask2 = (torch.abs(y2 - phi2) <= eps) & (dot_grad_phi2 > 1e-5)
    res2 = dot_grad_u2 - dot_grad_phi2
    j2 = torch.mean(res2[mask2]**2) if mask2.any() else torch.tensor(0.0, device=device)
    
    # ========== Tính J3 ==========
    phi3, _ = compute_phi_grad(X3_t)
    j3 = torch.mean(torch.relu(phi3 - y3)**2)
    
    # ========== Total loss ==========
    loss = j1 + j2 + 2.0 * j3
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record history
    history['j1'].append(j1.item())
    history['j2'].append(j2.item())
    history['j3'].append(j3.item())
    history['total'].append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | J1: {j1.item():.2e} | J2: {j2.item():.2e} | J3: {j3.item():.2e} | eps: {eps:.6f}")

print("-" * 60)
print("Training completed!")

# ===================================================================
# 8. VẼ KẾT QUẢ
# ===================================================================
def plot_results():
    res = 300
    x = np.linspace(0, 4, res)
    y = np.linspace(0, 4, res)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Predict
    device = next(net.parameters()).device
    pts_t = torch.from_numpy(pts).float().to(device)
    
    with torch.no_grad():
        u_pred = net(pts_t).cpu().numpy().reshape(res, res)
    
    # Tính φ cho cả bốn hình tròn
    # φ(x) = max(R_i - |x-c_i|)
    phi_total = np.zeros_like(X)
    for i in range(len(CENTERS)):
        dist = np.sqrt((X-CENTERS[i,0])**2 + (Y-CENTERS[i,1])**2)
        phi_i = RADII[i] - dist
        phi_total = np.maximum(phi_total, phi_i)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Hình 1: Nghiệm
    ax = axes[0]
    cf = ax.contourf(X, Y, u_pred, levels=50, cmap='viridis', alpha=0.9)
    plt.colorbar(cf, ax=ax, label='u(x)')
    
    # Vẽ u=0
    if u_pred.min() < 0 < u_pred.max():
        c0 = ax.contour(X, Y, u_pred, levels=[0], colors='red', linewidths=2.5, linestyles='-')
        ax.text(2, 2, 'u=0', color='white', fontweight='bold', fontsize=12,
               bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Vẽ biên vật cản φ=0
    ax.contour(X, Y, phi_total, levels=[0], colors='yellow', linewidths=2.5, linestyles='-')
    
    # Vẽ bốn hình tròn (đường nét đứt cyan)
    theta = np.linspace(0, 2*np.pi, 100)
    colors = ['c', 'm', 'orange', 'lime']
    for i in range(len(CENTERS)):
        x_circle = CENTERS[i,0] + RADII[i] * np.cos(theta)
        y_circle = CENTERS[i,1] + RADII[i] * np.sin(theta)
        ax.plot(x_circle, y_circle, color=colors[i], linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Circle {i+1}' if i < 2 else "")
        
        # Đánh dấu tâm các hình tròn
        ax.plot(CENTERS[i,0], CENTERS[i,1], 'o', color=colors[i], markersize=6, alpha=0.7)
    
    # Điểm quan sát
    ax.scatter(X_STAR[0,0], X_STAR[0,1], color='gold', marker='*', s=400, edgecolor='white', linewidth=2,
              label='Observer x*', zorder=10, alpha=0.95)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('DeepXDE - Four Circular Obstacles', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Hình 2: Loss history
    ax2 = axes[1]
    epochs = np.arange(len(history['j1']))
    
    ax2.semilogy(epochs, history['j1'], 'b-', alpha=0.3, linewidth=1, label='J1')
    ax2.semilogy(epochs, history['j2'], 'g-', alpha=0.3, linewidth=1, label='J2')
    ax2.semilogy(epochs, history['j3'], 'r-', alpha=0.3, linewidth=1, label='J3')
    ax2.semilogy(epochs, history['total'], 'k-', alpha=0.3, linewidth=1, label='Total')
    
    window = min(100, len(epochs)//10)
    if len(epochs) > window:
        def ma(data): return np.convolve(data, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        ax2.semilogy(ma_epochs, ma(history['j1']), 'b-', linewidth=2.5, label='J1 (MA)')
        ax2.semilogy(ma_epochs, ma(history['j2']), 'g-', linewidth=2.5, label='J2 (MA)')
        ax2.semilogy(ma_epochs, ma(history['j3']), 'r-', linewidth=2.5, label='J3 (MA)')
        ax2.semilogy(ma_epochs, ma(history['total']), 'k--', linewidth=3, label='Total (MA)')
    
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
    
    # In kết quả tốt nhất
    print(f"\nBest results at epoch {best_epoch}:")
    print(f"  Loss: {best_loss:.2e}")
    print(f"  J1: {history['j1'][best_epoch]:.2e}")
    print(f"  J2: {history['j2'][best_epoch]:.2e}")
    print(f"  J3: {history['j3'][best_epoch]:.2e}")

plot_results()