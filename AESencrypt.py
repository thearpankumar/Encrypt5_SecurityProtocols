import ctypes
import os
import secrets

# Load the shared library
lib = ctypes.cdll.LoadLibrary('./build/libaesencrypt.so')  # Ensure the shared library is compiled and available

# Define the aes_block structure
class aes_block(ctypes.Structure):
    _fields_ = [("block", ctypes.c_ubyte * 16)]

# Define the function prototype
lib.encrypt_cuda.argtypes = [
    ctypes.POINTER(aes_block),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int
]

# Function to load data from a file
def load_data_from_file(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

# Function to generate a random key and save it to a file
def generate_and_save_key(key_file_path, key_length=16):
    key = secrets.token_bytes(key_length)  # Generate a secure random key
    with open(key_file_path, 'wb') as key_file:
        key_file.write(key)
    return key

# Function to encrypt data
def encrypt_data(data, key):
    block_number = len(data) // 16
    if len(data) % 16 != 0:
        block_number += 1

    # Prepare the data
    aes_blocks = (aes_block * block_number)()
    for i in range(block_number):
        for j in range(16):
            idx = i * 16 + j
            if idx < len(data):
                aes_blocks[i].block[j] = data[idx]
            else:
                aes_blocks[i].block[j] = 0

    # Prepare the key
    key_array = (ctypes.c_ubyte * len(key))(*key)

    # Call the CUDA encryption function
    lib.encrypt_cuda(aes_blocks, key_array, len(key), block_number)

    # Retrieve the encrypted data
    encrypted_data = bytearray()
    for i in range(block_number):
        for j in range(16):
            encrypted_data.append(aes_blocks[i].block[j])

    return bytes(encrypted_data)

# Function to save encrypted data to a file
def save_encrypted_data(file_path, encrypted_data):
    with open(file_path, 'wb') as file:
        file.write(encrypted_data)

# Main function
def main():
    # File paths
    input_file_path = 'large_file.txt'
    key_file_path = 'key.txt'
    encrypted_file_path = 'encrypted.bin'

    # Load data from the input file
    data = load_data_from_file(input_file_path)
    print(f"Loaded data from {input_file_path}")

    # Generate and save the key
    key = generate_and_save_key(key_file_path)
    print(f"Generated and saved key to {key_file_path}")

    # Encrypt the data
    encrypted_data = encrypt_data(data, key)
    print("Data encrypted successfully")

    # Save the encrypted data to a file
    save_encrypted_data(encrypted_file_path, encrypted_data)
    print(f"Encrypted data saved to {encrypted_file_path}")


if __name__ == "__main__":
    main()