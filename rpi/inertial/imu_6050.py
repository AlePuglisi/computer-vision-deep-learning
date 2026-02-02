import smbus2
import time
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

bus = smbus2.SMBus(1)
MPU6050_ADDR = 0x68

# Wake up MPU6050
bus.write_byte_data(MPU6050_ADDR, 0x6B, 0x00)
time.sleep(0.1)

# Configure MPU6050
bus.write_byte_data(MPU6050_ADDR, 0x1A, 0x03)  # Low pass filter
bus.write_byte_data(MPU6050_ADDR, 0x1B, 0x18)  # Gyro ±2000°/s
bus.write_byte_data(MPU6050_ADDR, 0x1C, 0x10)  # Accel ±8g

ACCEL_SCALE = 4096.0
GYRO_SCALE = 16.4

def read_raw_data(addr, reg):
    """Read raw 16-bit data from sensor"""
    data = bus.read_i2c_block_data(addr, reg, 6)
    x = (data[0] << 8) | data[1]
    y = (data[2] << 8) | data[3]
    z = (data[4] << 8) | data[5]
    
    if x > 32767: x -= 65536
    if y > 32767: y -= 65536
    if z > 32767: z -= 65536
    
    return x, y, z

def read_accel():
    """Read accelerometer data in g"""
    x, y, z = read_raw_data(MPU6050_ADDR, 0x3B)
    return x/ACCEL_SCALE, y/ACCEL_SCALE, z/ACCEL_SCALE

def read_gyro():
    """Read gyroscope data in degrees/second"""
    x, y, z = read_raw_data(MPU6050_ADDR, 0x43)
    return x/GYRO_SCALE, y/GYRO_SCALE, z/GYRO_SCALE

def read_temperature():
    """Read temperature in Celsius with calibration"""
    data = bus.read_i2c_block_data(MPU6050_ADDR, 0x41, 2)
    temp_raw = (data[0] << 8) | data[1]
    if temp_raw > 32767:
        temp_raw -= 65536
    
    # Try this formula first
    temp_c = (temp_raw / 340.0) + 36.53
    
    # Calibration offset - adjust based on actual room temperature
    # If room is 25°C and sensor reads 40°C, offset is -15
    temp_offset = -15.0  # ADJUST THIS VALUE
    
    return temp_c + temp_offset

def calculate_angles(accel):
    """Calculate roll and pitch from accelerometer"""
    ax, ay, az = accel
    roll = math.atan2(ay, az) * (180.0 / math.pi)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az)) * (180.0 / math.pi)
    return roll, pitch

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.last_time = time.time()
    
    def update(self, accel, gyro):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        roll_acc, pitch_acc = calculate_angles(accel)
        
        self.angle_x += gyro[0] * dt
        self.angle_y += gyro[1] * dt
        self.angle_z += gyro[2] * dt
        
        self.angle_x = self.alpha * self.angle_x + (1 - self.alpha) * roll_acc
        self.angle_y = self.alpha * self.angle_y + (1 - self.alpha) * pitch_acc
        
        return self.angle_x, self.angle_y, self.angle_z

# Initialize data storage (keep last 100 points)
MAX_POINTS = 100
times = deque(maxlen=MAX_POINTS)
accel_x = deque(maxlen=MAX_POINTS)
accel_y = deque(maxlen=MAX_POINTS)
accel_z = deque(maxlen=MAX_POINTS)
gyro_x = deque(maxlen=MAX_POINTS)
gyro_y = deque(maxlen=MAX_POINTS)
gyro_z = deque(maxlen=MAX_POINTS)

comp_filter = ComplementaryFilter()
velocity = [0.0, 0.0, 0.0]

# Gyro bias calibration
print("Calibrating gyroscope... Keep sensor still!")
gyro_bias = [0.0, 0.0, 0.0]
samples = 100
for i in range(samples):
    gx, gy, gz = read_gyro()
    gyro_bias[0] += gx
    gyro_bias[1] += gy
    gyro_bias[2] += gz
    time.sleep(0.01)

gyro_bias[0] /= samples
gyro_bias[1] /= samples
gyro_bias[2] /= samples
print(f"Gyro bias: X={gyro_bias[0]:.2f} Y={gyro_bias[1]:.2f} Z={gyro_bias[2]:.2f} °/s")

accel_threshold = 0.05

# Setup matplotlib figure
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('IMU-Gyro Real-Time Data', fontsize=16, fontweight='bold')

# Accelerometer subplot
line_ax, = ax1.plot([], [], 'r-', label='Accel X', linewidth=2)
line_ay, = ax1.plot([], [], 'g-', label='Accel Y', linewidth=2)
line_az, = ax1.plot([], [], 'b-', label='Accel Z', linewidth=2)
ax1.set_ylabel('Acceleration (g)', fontsize=12)
ax1.set_ylim(-2, 2)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Gyroscope subplot
line_gx, = ax2.plot([], [], 'r-', label='Gyro X', linewidth=2)
line_gy, = ax2.plot([], [], 'g-', label='Gyro Y', linewidth=2)
line_gz, = ax2.plot([], [], 'b-', label='Gyro Z', linewidth=2)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Angular Velocity (°/s)', fontsize=12)
ax2.set_ylim(-200, 200)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

first_read_time = time.time()

def update_plot(frame):
    """Animation function called by matplotlib"""
    global velocity
    
    try:
        accel = read_accel()
        gyro = read_gyro()
        temp = read_temperature()
        current_time = time.time() - first_read_time
        
        # Remove gyro bias
        gyro = (gyro[0] - gyro_bias[0], 
                gyro[1] - gyro_bias[1], 
                gyro[2] - gyro_bias[2])
        
        roll, pitch, yaw = comp_filter.update(accel, gyro)
        
        # Velocity integration
        dt = 0.1
        accel_no_gravity = (accel[0], accel[1], accel[2] - 1.0)
        
        accel_magnitude = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)
        gyro_magnitude = math.sqrt(gyro[0]**2 + gyro[1]**2 + gyro[2]**2)
        
        if abs(accel_magnitude - 1.0) < accel_threshold and gyro_magnitude < 2.0:
            velocity = [0.0, 0.0, 0.0]
        else:
            velocity[0] += accel_no_gravity[0] * 9.81 * dt
            velocity[1] += accel_no_gravity[1] * 9.81 * dt
            velocity[2] += accel_no_gravity[2] * 9.81 * dt
            velocity[0] *= 0.98
            velocity[1] *= 0.98
            velocity[2] *= 0.98
        
        # Store data for plotting
        times.append(current_time)
        accel_x.append(accel[0])
        accel_y.append(accel[1])
        accel_z.append(accel[2])
        gyro_x.append(gyro[0])
        gyro_y.append(gyro[1])
        gyro_z.append(gyro[2])
        
        # Update plot data
        line_ax.set_data(times, accel_x)
        line_ay.set_data(times, accel_y)
        line_az.set_data(times, accel_z)
        line_gx.set_data(times, gyro_x)
        line_gy.set_data(times, gyro_y)
        line_gz.set_data(times, gyro_z)
        
        # Auto-scale x-axis
        if len(times) > 0:
            ax1.set_xlim(max(0, times[-1] - 10), times[-1] + 0.5)
            ax2.set_xlim(max(0, times[-1] - 10), times[-1] + 0.5)
        
        # Print to console
        print(f"\rTime: {current_time:.2f}s | "
              f"Accel: X={accel[0]:6.3f} Y={accel[1]:6.3f} Z={accel[2]:6.3f} g | "
              f"Gyro: X={gyro[0]:6.1f} Y={gyro[1]:6.1f} Z={gyro[2]:6.1f} °/s | "
              f"Temp: {temp:.1f}°C", end='')
        
    except Exception as e:
        print(f"\nError reading sensor: {e}")
    
    return line_ax, line_ay, line_az, line_gx, line_gy, line_gz

# Create animation - update every 100ms
ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

print("\n" + "="*80)
print("MPU-6050 Real-Time Plotting")
print("Close the plot window to stop")
print("="*80 + "\n")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    bus.close()