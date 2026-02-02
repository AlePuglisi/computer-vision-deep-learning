import smbus2
import time
import math

bus = smbus2.SMBus(1)
MPU9250_ADDR = 0x68
AK8963_ADDR = 0x0C  # Magnetometer address

# Wake up MPU9250
bus.write_byte_data(MPU9250_ADDR, 0x6B, 0x00)
time.sleep(0.1)

# Configure MPU9250
bus.write_byte_data(MPU9250_ADDR, 0x1A, 0x03)  # Set low pass filter
bus.write_byte_data(MPU9250_ADDR, 0x1B, 0x18)  # Gyro full scale ±2000°/s
bus.write_byte_data(MPU9250_ADDR, 0x1C, 0x10)  # Accel full scale ±8g

# Enable magnetometer bypass mode
bus.write_byte_data(MPU9250_ADDR, 0x37, 0x02)  # INT_PIN_CFG
time.sleep(0.1)

# Configure magnetometer
try:
    # Reset magnetometer
    bus.write_byte_data(AK8963_ADDR, 0x0A, 0x00)
    time.sleep(0.1)
    # Set to continuous measurement mode (16-bit, 100Hz)
    bus.write_byte_data(AK8963_ADDR, 0x0A, 0x16)
    time.sleep(0.1)
    mag_enabled = True
    print("Magnetometer initialized successfully")
except Exception as e:
    print(f"Magnetometer initialization failed: {e}")
    mag_enabled = False

# Calibration values (adjust these for your sensor)
ACCEL_SCALE = 1.0 
GYRO_SCALE = 1.0    
MAG_SCALE = 1.0      

def read_raw_data(addr, reg):
    """Read raw 16-bit data from sensor"""
    data = bus.read_i2c_block_data(addr, reg, 6)
    x = (data[0] << 8) | data[1]
    y = (data[2] << 8) | data[3]
    z = (data[4] << 8) | data[5]
    
    # Convert to signed
    if x > 32767: x -= 65536
    if y > 32767: y -= 65536
    if z > 32767: z -= 65536
    
    return x, y, z

def read_accel():
    """Read accelerometer data in g"""
    x, y, z = read_raw_data(MPU9250_ADDR, 0x3B)
    return x/ACCEL_SCALE, y/ACCEL_SCALE, z/ACCEL_SCALE

def read_gyro():
    """Read gyroscope data in degrees/second"""
    x, y, z = read_raw_data(MPU9250_ADDR, 0x43)
    return x/GYRO_SCALE, y/GYRO_SCALE, z/GYRO_SCALE

def read_mag():
    """Read magnetometer data in µT (microtesla)"""
    if not mag_enabled:
        return None, None, None
    
    try:
        # Check data ready
        status = bus.read_byte_data(AK8963_ADDR, 0x02)
        if status & 0x01:
            # Read magnetometer data
            data = bus.read_i2c_block_data(AK8963_ADDR, 0x03, 7)
            
            x = (data[1] << 8) | data[0]
            y = (data[3] << 8) | data[2]
            z = (data[5] << 8) | data[4]
            
            # Convert to signed
            if x > 32767: x -= 65536
            if y > 32767: y -= 65536
            if z > 32767: z -= 65536
            
            # Read status register 2 to complete reading
            st2 = data[6]
            
            return x*MAG_SCALE, y*MAG_SCALE, z*MAG_SCALE
    except Exception as e:
        print(f"Magnetometer read error: {e}")
    
    return None, None, None

def read_temperature():
    """Read temperature in Celsius"""
    data = bus.read_i2c_block_data(MPU9250_ADDR, 0x41, 2)
    temp_raw = (data[0] << 8) | data[1]
    if temp_raw > 32767:
        temp_raw -= 65536
    temp_c = (temp_raw / 333.87) + 21.0
    return temp_c

# For calculating velocity/position from accelerometer (integration)
# velocity = [0.0, 0.0, 0.0]
# position = [0.0, 0.0, 0.0]
# last_time = time.time()

print("Starting MPU9250 sensor reading...")
print("Accel (g), Gyro (°/s), Mag (µT), Temp (°C)")
print("-" * 80)

try:
    while True:
        # current_time = time.time()
        # dt = current_time - last_time
        # last_time = current_time
        
        # Read all sensors
        accel = read_accel()
        gyro = read_gyro()
        mag = read_mag()
        temp = read_temperature()
        
        # Calculate velocity (simple integration - not very accurate without proper filtering)
        # Remove gravity component (assuming Z-axis is vertical)
        accel_no_gravity = (accel[0], accel[1], accel[2] - 1.0)
        
        
        # Print data
        print(f"Accel: X={accel[0]:7.3f} Y={accel[1]:7.3f} Z={accel[2]:7.3f} g")
        print(f"Gyro:  X={gyro[0]:7.1f} Y={gyro[1]:7.1f} Z={gyro[2]:7.1f} °/s")
        
        if mag[0] is not None:
            print(f"Mag:   X={mag[0]:7.1f} Y={mag[1]:7.1f} Z={mag[2]:7.1f} µT")
            # Calculate heading (compass direction)
            heading = math.atan2(mag[1], mag[0]) * (180.0 / math.pi)
            if heading < 0:
                heading += 360
            print(f"Heading: {heading:.1f}°")
        
        print(f"Temp:  {temp:.1f}°C")
        # print(f"Vel:   X={velocity[0]:7.2f} Y={velocity[1]:7.2f} Z={velocity[2]:7.2f} m/s")
        print("-" * 80)
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopped by user")
    bus.close()