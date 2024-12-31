# CUDA AES Encryption/Decryption with Padding

This project provides CUDA-based implementations of AES encryption and decryption, along with padding support. The code is designed to be compiled into shared libraries (`*.so` files) and can be used in Python via `ctypes`.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Build Instructions](#build-instructions)
4. [File Structure](#file-structure)

---

## Project Overview

This project includes the following components:

- **AES Encryption/Decryption**:
  - `Aesencrypt.cu`: CUDA implementation of AES encryption.
  - `Aesdecrypt.cu`: CUDA implementation of AES decryption.

- **Padding Encryption/Decryption**:
  - `encrypt.cu`: CUDA implementation of padding encryption.
  - `decrypt.cu`: CUDA implementation of padding decryption.

The compiled shared libraries are placed in the `build` directory for better organization.

---

## Dependencies

To build and use this project, you need the following:

- **CUDA Toolkit**: Ensure CUDA is installed on your system. You can download it from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit).
- **Python**: Python 3.x is required for using the libraries via `ctypes`.
- **Make**: The project uses a `Makefile` for building the libraries.

---

## Build Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   make


.
├── Aesencrypt.cu          # AES encryption CUDA source
├── Aesdecrypt.cu          # AES decryption CUDA source
├── encrypt.cu             # Padding encryption CUDA source
├── decrypt.cu             # Padding decryption CUDA source
├── Makefile               # Build script
├── README.md              # Project documentation
├── build/                 # Directory for compiled shared libraries
│   ├── libaesencrypt.so   # AES encryption library
│   ├── libaesdecrypt.so   # AES decryption library
│   ├── padlibencrypt.so   # Padding encryption library
│   └── padlibdecrypt.so   # Padding decryption library
└── main.py                # Example Python script
