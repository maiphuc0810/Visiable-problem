import deepxde as dde
import numpy as np
import tensorflow as tf

# ===================================================================
# 1. THÔNG SỐ BÀI TOÁN & HÀM PHỤ TRỢ
# ===================================================================
X_STAR = np.array([[0.0, 0.0]], dtype=np.float32)
CENTER = np.array([[2.0, 2.0]], dtype=np.float32)
R = 1.0
# phi(x*) = 1 - ||(0,0) - (2,2)|| = 1 - sqrt(8)
PHI_X_STAR = 1.0 - np.sqrt(8.0)

def get_phi(x):
    """Hàm vật cản phi(x) = 1 - ||x - c||"""
    # Sử dụng tf để tính toán trong đồ thị của DeepXDE
    dist = tf.norm(x - CENTER, axis=1, keepdims=True)
    return R - dist

def get_grad_phi(x):
    """Gradient của phi: grad_phi = -(x-c)/||x-c||"""
    diff = x - CENTER
    dist = tf.norm(diff, axis=1, keepdims=True) + 1e-12
    return -diff / dist

# Biến toàn cục để cập nhật epsilon theo Algorithm 1
epsilon_v = tf.Variable(0.1, trainable=False, dtype=tf.float32)

# ===================================================================
# 2. ĐỊNH NGHĨA PDE THEO ALGORITHM 1 (3.17 & 3.19)
# ===================================================================
def pde(x, y):
    """
    Triển khai chính xác Algorithm 1 - Tính toán đạo hàm theo từng thành phần
    """
    # 1. Tính gradient của u (xi_N) theo từng thành phần x1 và x2
    # i=0 vì đầu ra u chỉ có 1 chiều. j=0 là x1, j=1 là x2
    du_dx1 = dde.grad.jacobian(y, x, i=0, j=0)
    du_dx2 = dde.grad.jacobian(y, x, i=0, j=1)
    
    # 2. Tính toán phi và gradient của phi
    phi = get_phi(x)
    # Tính gradient phi theo giải tích: grad_phi = -(x - center) / ||x - center||
    diff_center = x - CENTER
    dist = tf.norm(diff_center, axis=1, keepdims=True) + 1e-12
    grad_phi_x1 = -diff_center[:, 0:1] / dist
    grad_phi_x2 = -diff_center[:, 1:2] / dist
    
    # 3. Tính toán các tích vô hướng (Dot Products) theo Algorithm 1
    diff_x_star_x1 = x[:, 0:1] - X_STAR[0, 0]
    diff_x_star_x2 = x[:, 1:2] - X_STAR[0, 1]
    
    # (x - x*) . grad_u
    dot_grad_u = diff_x_star_x1 * du_dx1 + diff_x_star_x2 * du_dx2
    
    # (x - x*) . grad_phi
    dot_grad_phi = diff_x_star_x1 * grad_phi_x1 + diff_x_star_x2 * grad_phi_x2
    
    # (x - x*) . (grad_u - grad_phi)
    dot_diff_grad = dot_grad_u - dot_grad_phi

    # 4. Phân loại các tập hợp G1, G2, G3 bằng Logic Mask
    # G1: {u > phi} OR {(x - x*).grad_phi <= 0}
    mask_G1 = tf.logical_or(y > phi, dot_grad_phi <= 0)
    
    # G2: {|u - phi| <= eps} AND {(x - x*).grad_phi > 0}
    mask_G2 = tf.logical_and(tf.abs(y - phi) <= epsilon_v, dot_grad_phi > 0)
    
    # G3: {phi > u}
    mask_G3 = phi > y

    # 5. Tính Residual Jn theo (3.19)
    # Dùng tf.where để lọc các điểm thuộc đúng tập hợp
    res_G1 = tf.where(mask_G1, dot_grad_u, tf.zeros_like(dot_grad_u))
    res_G2 = tf.where(mask_G2, dot_diff_grad, tf.zeros_like(dot_diff_grad))
    res_G3 = tf.where(mask_G3, phi - y, tf.zeros_like(y))

    return [res_G1, res_G2, res_G3]

# ===================================================================
# 3. TRIAL SOLUTION TRANSFORMATION (3.18)
# ===================================================================
def output_transform(x, y):
    """
    Ép cấu trúc xi_N(x) = phi(x*) + |x - x*|^2 * NN(x)
    Đây là Hard Constraint giúp thỏa mãn u(x*) = phi(x*) tự động.
    """
    r2 = tf.reduce_sum(tf.square(x - X_STAR), axis=1, keepdims=True)
    return PHI_X_STAR + r2 * y

# ===================================================================
# 4. THIẾT LẬP MÔ HÌNH VÀ HUẤN LUYỆN
# ===================================================================
# Miền hình học [0, 4] x [0, 4]
geom = dde.geometry.Rectangle([0, 0], [4, 4])

# Định nghĩa dữ liệu PDE
# Lưu ý: num_boundary=0 vì chúng ta đã có Hard Constraint cho x*
data = dde.data.PDE(
    geom, pde, [], 
    num_domain=2000, 
    num_boundary=0, 
    num_test=500
)

# Kiến trúc mạng: 4 lớp ẩn, mỗi lớp 128 nodes, hàm kích hoạt tanh (theo Fig 11 PINNS.pdf)
layer_size = [2] + [128] * 4 + [1]
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# Callback để giảm epsilon theo thời gian (chuỗi {epsilon_n} -> 0)
def update_epsilon(obj):
    new_eps = max(epsilon_v.numpy() * 0.999, 1e-4)
    epsilon_v.assign(new_eps)

epsilon_callback = dde.callbacks.VariableValue(epsilon_v, period=100)

# Huấn luyện
model.compile("adam", lr=0.001, loss_weights=[1, 1, 1])
model.train(iterations=20000, callbacks=[epsilon_callback])

# Tối ưu hóa tinh chỉnh bằng L-BFGS
model.compile("L-BFGS")
model.train()

# ===================================================================
# 5. TRỰC QUAN HÓA KẾT QUẢ
# ===================================================================
import matplotlib.pyplot as plt

def plot_results():
    # 1. Tạo lưới điểm để dự đoán
    res = 200 # Tăng độ phân giải để đường contour mượt hơn
    x_grid = np.linspace(0, 4, res)
    y_grid = np.linspace(0, 4, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    
    # 2. Dự đoán giá trị từ mô hình PINN
    u_pred = model.predict(pts).reshape(res, res)
    phi_val = np.array([PHI_X_STAR if (p[0]==0 and p[1]==0) else 1.0 - np.linalg.norm(p - CENTER) for p in pts]).reshape(res, res)

    plt.figure(figsize=(10, 8))
    
    # 3. Vẽ nền màu (Filled Contour)
    # Vùng xanh là vùng có giá trị u cao, vùng vàng/tím thấp hơn
    cf = plt.contourf(X, Y, u_pred, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(cf, label='Visibility Value $u(x)$')
    
    # 4. Vẽ các đường mức (Contours)
    # Vẽ các đường mức mảnh để thấy sự thay đổi của hàm nghiệm
    contours = plt.contour(X, Y, u_pred, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    plt.clabel(contours, inline=True, fontsize=8) # Gắn nhãn giá trị lên đường contour

    # 5. VẼ ĐƯỜNG NGHIỆM BẰNG 0 (Zero Level Set)
    # Đây thường là ranh giới quan trọng
    line_zero = plt.contour(X, Y, u_pred, levels=[0], colors='red', linewidths=2, linestyles='--')
    plt.clabel(line_zero, fmt='u=0', inline=True, fontsize=12)

    # 6. Vẽ ranh giới vật cản và điểm quan sát
    circle = plt.Circle((2, 2), 1, color='black', fill=False, linewidth=2, label='Obstacle Boundary')
    plt.gca().add_patch(circle)

    plt.scatter(0, 0, color='red', marker='*', s=250, label='Observer $x^*$', zorder=5)
    
    plt.title("PINN Visibility Solution with Contours and $u=0$ line")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.gca().set_aspect('equal')
    plt.show()

# Gọi hàm vẽ sau khi train xong
plot_results()