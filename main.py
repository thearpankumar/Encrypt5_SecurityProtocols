import ctypes

# Load the shared library
lib = ctypes.CDLL('./build/padlibencrypt.so')

# Define the argument types for the encrypt_files function
lib.encrypt_files.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

def encrypt_files(input_file, encrypted_file, key_file):
    # Call the CUDA function
    lib.encrypt_files(input_file.encode(), encrypted_file.encode(), key_file.encode())

# Load the shared library
lib2 = ctypes.CDLL('./build/padlibdecrypt.so')

# Define the argument types for the decrypt_files function
lib2.decrypt_files.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

def decrypt_files(encrypted_file, key_file, decrypted_file):
    # Call the CUDA function
    lib2.decrypt_files(encrypted_file.encode(), key_file.encode(), decrypted_file.encode())


# Example usage
if __name__ == "__main__":
    input_file = "large_file.txt"
    encrypted_file = "encrypted.txt"
    key_file = "key.txt"

    encrypt_files(input_file, encrypted_file, key_file)
    print("Encryption completed!")

    decrypted_file = "decrypted.txt"

    decrypt_files(encrypted_file, key_file, decrypted_file)
    print("Decryption completed!")