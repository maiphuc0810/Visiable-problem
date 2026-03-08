import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa các tham số
Omega = [0, 4, 0, 4]  # [x_min, x_max, y_min, y_max]
obstacle_center = (2, 2)
obstacle_radius = 1
x_star = (0, 0)  # Điểm quan sát

# Định nghĩa hàm chướng ngại
def phi(x, y):
    return 1 - (x - 2)**2 - (y - 2)**2

# Hàm tính nghiệm nhớt ψ(x)
def psi(x, y, n_samples=1000):
    """Tính ψ(x) = max_{t∈[0,1]} φ(x* + t(x - x*))"""
    # Vector từ x* đến (x,y)
    if x == x_star[0] and y == x_star[1]:
        return phi(x, y)
    
    # Lấy mẫu các điểm trên đoạn thẳng
    t_values = np.linspace(0, 1, n_samples)
    max_value = -np.inf
    
    for t in t_values:
        # Điểm trên đoạn thẳng: x* + t*(x - x*)
        x_t = x_star[0] + t * (x - x_star[0])
        y_t = x_star[1] + t * (y - x_star[1])
        phi_val = phi(x_t, y_t)
        if phi_val > max_value:
            max_value = phi_val
    
    return max_value

# Tạo lưới điểm
N = 200
x = np.linspace(Omega[0], Omega[1], N)
y = np.linspace(Omega[2], Omega[3], N)
X, Y = np.meshgrid(x, y)

# Tính giá trị ψ và xác định vùng nhìn thấy
Psi = np.zeros_like(X)
visible = np.zeros_like(X, dtype=bool)

for i in range(N):
    for j in range(N):
        Psi[i, j] = psi(X[i, j], Y[i, j])
        # Điểm nhìn thấy nếu ψ(x) < 0 (có thể ≤ 0 tùy định nghĩa)
        # Ở đây dùng < 0 để không tính biên vật cản là nhìn thấy
        visible[i, j] = (Psi[i, j] < 0)

# Vẽ hình
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Hình 1: Miền Ω, vật cản và điểm quan sát
ax1 = axes[0]
ax1.set_xlim(Omega[0], Omega[1])
ax1.set_ylim(Omega[2], Omega[3])
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Miền Ω, vật cản và điểm quan sát')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Vẽ vật cản (hình tròn)
circle = plt.Circle(obstacle_center, obstacle_radius, 
                    color='red', alpha=0.6, label='Vật cản')
ax1.add_patch(circle)

# Vẽ điểm quan sát
ax1.scatter([x_star[0]], [x_star[1]], color='blue', s=100, 
            label=f'Điểm quan sát {x_star}', zorder=5)
ax1.legend()

# Hình 2: Vùng nhìn thấy theo nghiệm nhớt ψ
ax2 = axes[1]
ax2.set_xlim(Omega[0], Omega[1])
ax2.set_ylim(Omega[2], Omega[3])
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('Vùng nhìn thấy từ nghiệm nhớt ψ')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Tạo mask cho các vùng
# Vật cản
obstacle_mask = (X - 2)**2 + (Y - 2)**2 <= 1

# Vùng nhìn thấy (ψ(x) < 0 và không nằm trong vật cản)
visible_mask = visible & (~obstacle_mask)

# Vùng không nhìn thấy (ψ(x) ≥ 0 và không nằm trong vật cản)
not_visible_mask = (~visible) & (~obstacle_mask)

# Vẽ các vùng
ax2.contourf(X, Y, obstacle_mask, levels=[0.5, 1.5], 
             colors=['red'], alpha=0.6, label='Vật cản')
ax2.contourf(X, Y, visible_mask, levels=[0.5, 1.5], 
             colors=['green'], alpha=0.4, label='Nhìn thấy được')
ax2.contourf(X, Y, not_visible_mask, levels=[0.5, 1.5], 
             colors=['gray'], alpha=0.4, label='Không nhìn thấy được')

# Vẽ một số contour của ψ
psi_levels = np.linspace(Psi.min(), Psi.max(), 20)
CS = ax2.contour(X, Y, Psi, levels=psi_levels, colors='blue', alpha=0.3, linewidths=0.5)
ax2.clabel(CS, inline=1, fontsize=8, fmt='%1.1f')

# Vẽ contour ψ = 0 (biên vùng nhìn thấy)
ax2.contour(X, Y, Psi, levels=[0], colors='black', linewidths=2, label='ψ = 0')

# Vẽ điểm quan sát
ax2.scatter([x_star[0]], [x_star[1]], color='blue', s=100, 
            label=f'Điểm quan sát {x_star}', zorder=5)

ax2.legend()
plt.tight_layout()
plt.show()

# Tính phần trăm diện tích nhìn thấy
total_points = N * N
obstacle_points = np.sum(obstacle_mask)
visible_points = np.sum(visible_mask)
not_visible_points = np.sum(not_visible_mask)

print("=== KẾT QUẢ THEO NGHIỆM NHỚT ψ ===")
print(f"Tổng số điểm lưới: {total_points}")
print(f"Số điểm trong vật cản: {obstacle_points} ({obstacle_points/total_points*100:.2f}%)")
print(f"Số điểm nhìn thấy được: {visible_points} ({visible_points/total_points*100:.2f}%)")
print(f"Số điểm không nhìn thấy được: {not_visible_points} ({not_visible_points/total_points*100:.2f}%)")
print(f"\nGiá trị ψ tại một số điểm đặc biệt:")
print(f"ψ(x*) = ψ{tuple(x_star)} = {psi(x_star[0], x_star[1]):.4f}")
print(f"ψ(trung tâm vật cản) = ψ(2,2) = {psi(2, 2):.4f}")
print(f"ψ(điểm đối diện) = ψ(4,4) = {psi(4, 4):.4f}")