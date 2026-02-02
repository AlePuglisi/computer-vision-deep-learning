import smbus2
import time
import math

bus = smbus2.SMBus(1)
MPU9250_ADDR = 0x68

# Wake up MPU9250
bus.write_byte_data(MPU9250_ADDR, 0x6B, 0x00)
time.sleep(0.1)

# Configure MPU9250
bus.write_byte_data(MPU9250_ADDR, 0x1A, 0x03)
bus.write_byte_data(MPU9250_ADDR, 0x1B, 0x18)
bus.write_byte_data(MPU9250_ADDR, 0x1C, 0x10)

ACCEL_SCALE = 4096.0
GYRO_SCALE = 16.4
MAG_SCALE = 0.15

def setup_mag_aux_v2():
    """Alternative magnetometer setup for GY-9250"""
    try:
        # Disable I2C master first
        bus.write_byte_data(MPU9250_ADDR, 0x6A, 0x00)
        time.sleep(0.1)
        
        # Enable I2C master mode
        bus.write_byte_data(MPU9250_ADDR, 0x6A, 0x20)
        time.sleep(0.1)
        
        # Set I2C master clock to 400kHz
        bus.write_byte_data(MPU9250_ADDR, 0x24, 0x0D)
        time.sleep(0.1)
        
        # Power down magnetometer first
        # Write 0x00 to AK8963 CNTL1 register (0x0A)
        bus.write_byte_data(MPU9250_ADDR, 0x25, 0x0C)  # I2C_SLV0_ADDR (write)
        bus.write_byte_data(MPU9250_ADDR, 0x26, 0x0A)  # I2C_SLV0_REG
        bus.write_byte_data(MPU9250_ADDR, 0x63, 0x00)  # I2C_SLV0_DO (power down)
        bus.write_byte_data(MPU9250_ADDR, 0x27, 0x81)  # I2C_SLV0_CTRL (enable)
        time.sleep(0.2)
        
        # Read WHO_AM_I from magnetometer
        bus.write_byte_data(MPU9250_ADDR, 0x25, 0x0C | 0x80)  # Read mode
        bus.write_byte_data(MPU9250_ADDR, 0x26, 0x00)  # WHO_AM_I register
        bus.write_byte_data(MPU9250_ADDR, 0x27, 0x81)  # Enable, 1 byte
        time.sleep(0.1)
        
        who_am_i = bus.read_byte_data(MPU9250_ADDR, 0x49)
        print(f"Magnetometer WHO_AM_I: 0x{who_am_i:02X}")
        
        if who_am_i == 0x00 or who_am_i == 0xFF:
            print("Magnetometer not responding - trying bypass mode")
            return setup_mag_bypass()
        
        if who_am_i != 0x48:
            print(f"Unexpected WHO_AM_I: 0x{who_am_i:02X}")
            return False
        
        # Set to continuous mode 2 (100Hz, 16-bit)
        bus.write_byte_data(MPU9250_ADDR, 0x25, 0x0C)  # Write mode
        bus.write_byte_data(MPU9250_ADDR, 0x26, 0x0A)  # CNTL1
        bus.write_byte_data(MPU9250_ADDR, 0x63, 0x16)  # Continuous mode 2
        bus.write_byte_data(MPU9250_ADDR, 0x27, 0x81)
        time.sleep(0.2)
        
        # Configure to read magnetometer data continuously
        bus.write_byte_data(MPU9250_ADDR, 0x25, 0x0C | 0x80)  # Read mode
        bus.write_byte_data(MPU9250_ADDR, 0x26, 0x03)  # Start from HXL
        bus.write_byte_data(MPU9250_ADDR, 0x27, 0x87)  # Enable, 7 bytes
        time.sleep(0.1)
        
        print("✓ Magnetometer initialized via auxiliary I2C")
        return True
        
    except Exception as e:
        print(f"✗ Auxiliary I2C failed: {e}")
        return False

def setup_mag_bypass():
    """Try bypass mode as fallback"""
    try:
        # Disable I2C master
        bus.write_byte_data(MPU9250_ADDR, 0x6A, 0x00)
        time.sleep(0.1)
        
        # Enable bypass mode
        bus.write_byte_data(MPU9250_ADDR, 0x37, 0x02)
        time.sleep(0.1)
        
        # Try to access magnetometer directly
        AK8963_ADDR = 0x0C
        who_am_i = bus.read_byte_data(AK8963_ADDR, 0x00)
        print(f"Magnetometer WHO_AM_I (bypass): 0x{who_am_i:02X}")
        
        if who_am_i != 0x48:
            print("Bypass mode also failed")
            return False
        
        # Power down
        bus.write_byte_data(AK8963_ADDR, 0x0A, 0x00)
        time.sleep(0.1)
        
        # Set to continuous mode
        bus.write_byte_data(AK8963_ADDR, 0x0A, 0x16)
        time.sleep(0.1)
        
        print("Magnetometer initialized via bypass mode")
        return True
        
    except Exception as e:
        print(f"Bypass mode failed: {e}")
        return False

mag_enabled = setup_mag_aux_v2()
mag_bypass_mode = False  # Track which mode is active

def read_raw_data(addr, reg):
    data = bus.read_i2c_block_data(addr, reg, 6)
    x = (data[0] << 8) | data[1]
    y = (data[2] << 8) | data[3]
    z = (data[4] << 8) | data[5]
    
    if x > 32767: x -= 65536
    if y > 32767: y -= 65536
    if z > 32767: z -= 65536
    
    return x, y, z

def read_accel():
    x, y, z = read_raw_data(MPU9250_ADDR, 0x3B)
    return x/ACCEL_SCALE, y/ACCEL_SCALE, z/ACCEL_SCALE

def read_gyro():
    x, y, z = read_raw_data(MPU9250_ADDR, 0x43)
    return x/GYRO_SCALE, y/GYRO_SCALE, z/GYRO_SCALE

def read_mag():
    if not mag_enabled:
        return None, None, None
    
    try:
        if mag_bypass_mode:
            # Read directly from magnetometer
            AK8963_ADDR = 0x0C
            status = bus.read_byte_data(AK8963_ADDR, 0x02)
            if not (status & 0x01):
                return None, None, None
            
            data = bus.read_i2c_block_data(AK8963_ADDR, 0x03, 7)
        else:
            # Read from auxiliary I2C buffer
            data = bus.read_i2c_block_data(MPU9250_ADDR, 0x49, 7)
        
        x = (data[1] << 8) | data[0]
        y = (data[3] << 8) | data[2]
        z = (data[5] << 8) | data[4]
        
        if x > 32767: x -= 65536
        if y > 32767: y -= 65536
        if z > 32767: z -= 65536
        
        # Check overflow
        if data[6] & 0x08:
            return None, None, None
        
        return x*MAG_SCALE, y*MAG_SCALE, z*MAG_SCALE
        
    except:
        return None, None, None

def read_temperature():
    data = bus.read_i2c_block_data(MPU9250_ADDR, 0x41, 2)
    temp_raw = (data[0] << 8) | data[1]
    if temp_raw > 32767:
        temp_raw -= 65536
    return (temp_raw / 333.87) + 21.0

velocity = [0.0, 0.0, 0.0]
last_time = time.time()

print("\nGY-9250 Sensor Reading")
print("-" * 80)

try:
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        accel = read_accel()
        gyro = read_gyro()
        mag = read_mag()
        temp = read_temperature()
        
        accel_no_gravity = (accel[0], accel[1], accel[2] - 1.0)
        velocity[0] += accel_no_gravity[0] * 9.81 * dt
        velocity[1] += accel_no_gravity[1] * 9.81 * dt
        velocity[2] += accel_no_gravity[2] * 9.81 * dt
        
        print(f"Accel: X ={accel[0]:5.3f} Y ={accel[1]:5.3f} Z ={accel[2]:5.3f} g")
        print(f"Gyro:  X ={gyro[0]:5.3f} Y ={gyro[1]:5.3f} Z ={gyro[2]:5.3f} °/s")
        
        if mag[0] is not None:
            print(f"Mag:   X ={mag[0]:5.3f} Y ={mag[1]:5.3f} Z ={mag[2]:5.3f} µT")
            heading = math.atan2(mag[1], mag[0]) * (180.0 / math.pi)
            if heading < 0:
                heading += 360
            print(f"Heading: {heading:.1f}°")
        
        print(f"Temp:  T = {temp:.1f}°C")
        print(f"Vel:   X ={velocity[0]:5.3f} Y ={velocity[1]:5.3f} Z ={velocity[2]:5.3f} m/s")
        print("-" * 80)
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopped by user")
    bus.close()