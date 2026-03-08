import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===================================================================
# 1. THÔNG SỐ BÀI TOÁN
# ===================================================================
X_STAR = np.array([[0.0, 0.0]], dtype=np.float32)
CENTER = np.array([[2.0, 2.0]], dtype=np.float32)
R = 1.0
PHI_X_STAR = float(1.0 - np.sqrt(8.0))

EPS_INIT = 0.1
EPS_FINAL = 1e-4
# Sử dụng biến Global để Callback có thể cập nhật
epsilon_v = EPS_INIT 

# ===================================================================
# 2. HÌNH HỌC VÀ PDE
# ===================================================================
geom = dde.geometry.Rectangle([0, 0], [4, 4])

def compute_phi_grad(x):
    # Hàm bổ trợ tính phi và gradient của vật cản
    center_t = torch.tensor(CENTER, dtype=torch.float32, device=x.device)
    diff_c = x - center_t
    dist = torch.norm(diff_c, dim=1, keepdim=True) + 1e-8
    phi = R - dist
    grad_phi = -diff_c / dist
    return phi, grad_phi

def pde(x, y):
    # Tính gradient u theo x
    du_dx = dde.grad.jacobian(y, x, i=0)
    du_dx1, du_dx2 = du_dx[:, 0:1], du_dx[:, 1:2]
    
    phi, grad_phi = compute_phi_grad(x)
    
    # Vector (x - x*)
    x_star_t = torch.tensor(X_STAR, dtype=torch.float32, device=x.device)
    dx = x[:, 0:1] - x_star_t[:, 0:1]
    dy = x[:, 1:2] - x_star_t[:, 1:2]
    
    # Dot products theo công thức bài toán Visibility
    dot_grad_u = dx * du_dx1 + dy * du_dx2
    dot_grad_phi = dx * grad_phi[:, 0:1] + dy * grad_phi[:, 1:2]
    
    # Masks phân vùng G1, G2, G3
    # Lưu ý: epsilon_v lấy từ biến global đã được callback cập nhật
    mask_G1 = ((y > phi) | (dot_grad_phi <= 1e-5)).float()
    mask_G2 = ((torch.abs(y - phi) <= epsilon_v) & (dot_grad_phi > 1e-5)).float()
    mask_G3 = (phi > y).float()

    res_G1 = mask_G1 * dot_grad_u
    res_G2 = mask_G2 * (dot_grad_u - dot_grad_phi)
    res_G3 = mask_G3 * (phi - y)
    
    return [res_G1, res_G2, res_G3]

# ===================================================================
# 3. THIẾT LẬP DỮ LIỆU VÀ MẠNG
# ===================================================================
# Khởi tạo với 5000 điểm domain
data = dde.data.PDE(geom, pde, [], num_domain=5000, num_test=1000)

def output_transform(x, y):
    # Ép điều kiện u(x*) = phi(x*)
    x_star_t = torch.tensor(X_STAR, dtype=torch.float32, device=x.device)
    r2 = torch.sum(torch.square(x - x_star_t), dim=1, keepdim=True)
    return PHI_X_STAR + r2 * y

net = dde.nn.FNN([2] + [64] * 5 + [1], "gelu", "Glorot uniform")
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# ===================================================================
# 4. CALLBACKS: CẬP NHẬT EPSILON
# ===================================================================
class EpsilonCallback(dde.callbacks.Callback):
    def on_epoch_begin(self):
        global epsilon_v
        epoch = self.model.train_state.epoch
        epsilon_v = max(EPS_INIT * (0.9996 ** epoch), EPS_FINAL)

# ===================================================================
# 5. HUẤN LUYỆN VỚI CHIẾN THUẬT RAR
# ===================================================================

# Bước A: Huấn luyện sơ bộ với Adam
model.compile("adam", lr=1e-3, loss_weights=[1, 1, 10])
print("--- Training Stage 1: Adam ---")
model.train(iterations=10000, callbacks=[EpsilonCallback()])

# Bước B: Áp dụng RAR (Residual-based Adaptive Refinement)
# DeepXDE sẽ tự chọn các điểm có lỗi cao nhất để thêm vào tập train
print("--- Training Stage 2: RAR Refinement ---")
for i in range(5):
    # Lấy thêm 100 điểm có residual lớn nhất trong 2000 điểm ngẫu nhiên
    model.train(iterations=2000, callbacks=[EpsilonCallback()])
    
    # Thuật toán RAR thủ công tích hợp trong vòng lặp
    X = geom.random_points(2000)
    # Tính lỗi tại các điểm mới
    f = model.predict(X, operator=pde)
    err = np.mean([np.abs(fi) for fi in f], axis=0).flatten()
    # Chọn ra các điểm tệ nhất
    idx = np.argsort(err)[-100:]
    model.data.add_anchors(X[idx])
    print(f"RAR Cycle {i+1}: Added 100 points. Total points: {len(model.data.train_x_all)}")

# Bước C: Hội tụ cuối bằng L-BFGS
print("--- Training Stage 3: L-BFGS Fine-tuning ---")
model.compile("L-BFGS", loss_weights=[1, 1, 5])
losshistory, train_state = model.train()

# ===================================================================
# 6. HIỂN THỊ KẾT QUẢ
# ===================================================================
def plot_results():
    res = 200
    x = np.linspace(0, 4, res)
    y = np.linspace(0, 4, res)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    
    u_pred = model.predict(pts).reshape(res, res)
    
    plt.figure(figsize=(10, 8))
    cf = plt.contourf(X, Y, u_pred, levels=100, cmap='jet')
    plt.colorbar(cf, label='u(x)')
    
    # Vẽ vật cản (biên phi = 0)
    dist_c = np.sqrt((X-2)**2 + (Y-2)**2)
    phi = 1.0 - dist_c
    plt.contour(X, Y, phi, levels=[0], colors='white', linewidths=2)
    
    plt.scatter(0, 0, color='yellow', marker='*', s=200, label='Source (0,0)')
    plt.title("Solution with DeepXDE + RAR + L-BFGS")
    plt.legend()
    plt.show()

plot_results()