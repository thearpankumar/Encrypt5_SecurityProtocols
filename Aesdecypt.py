import ctypes
import numpy as np

# Load the shared library
lib = ctypes.cdll.LoadLibrary('./build/libaesdecrypt.so')  # Ensure the shared library is compiled and available

# Define the aes_block structure
class aes_block(ctypes.Structure):
    _fields_ = [("block", ctypes.c_ubyte * 16)]

# Define the function prototype
lib.decrypt_cuda.argtypes = [
    ctypes.POINTER(aes_block),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int
]

# Function to load encrypted data from a file
def load_encrypted_data(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

# Function to load the key from a file
def load_key(key_file_path):
    with open(key_file_path, 'rb') as key_file:
        return key_file.read()

# Function to decrypt data
def decrypt_data(encrypted_data, key):
    block_number = len(encrypted_data) // 16
    if len(encrypted_data) % 16 != 0:
        block_number += 1

    # Prepare the data
    aes_blocks = (aes_block * block_number)()
    for i in range(block_number):
        for j in range(16):
            idx = i * 16 + j
            if idx < len(encrypted_data):
                aes_blocks[i].block[j] = encrypted_data[idx]
            else:
                aes_blocks[i].block[j] = 0

    # Prepare the key
    key_array = (ctypes.c_ubyte * len(key))(*key)

    # Call the CUDA decryption function
    lib.decrypt_cuda(aes_blocks, key_array, len(key), block_number)

    # Retrieve the decrypted data
    decrypted_data = bytearray()
    for i in range(block_number):
        for j in range(16):
            decrypted_data.append(aes_blocks[i].block[j])

    return bytes(decrypted_data)

# Function to save decrypted data to a file
def save_decrypted_data(file_path, decrypted_data):
    with open(file_path, 'wb') as file:
        file.write(decrypted_data)

# Main function
def main():
    # File paths
    encrypted_file_path = 'encrypted.bin'
    key_file_path = 'key.txt'
    decrypted_file_path = 'decrypted.txt'

    # Load encrypted data
    encrypted_data = load_encrypted_data(encrypted_file_path)
    print(f"Loaded encrypted data from {encrypted_file_path}")

    # Load the key
    key = load_key(key_file_path)
    print(f"Loaded key from {key_file_path}")

    # Decrypt the data
    decrypted_data = decrypt_data(encrypted_data, key)
    print("Data decrypted successfully")

    # Save the decrypted data to a file
    save_decrypted_data(decrypted_file_path, decrypted_data)
    print(f"Decrypted data saved to {decrypted_file_path}")

if __name__ == "__main__":
    main()