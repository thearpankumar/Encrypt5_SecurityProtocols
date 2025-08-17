# Encrypt5_SecurityProtocols

![Version](https://img.shields.io/badge/version-5.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Rust](https://img.shields.io/badge/rust-1.73+-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-orange.svg)
![FastAPI](https://img.shields.io/badge/fastapi-latest-red.svg)
![Jupyter](https://img.shields.io/badge/jupyter-supported-blueviolet.svg)
![Azure](https://img.shields.io/badge/azure-cloud-informational.svg)
![P2P](https://img.shields.io/badge/p2p-networking-important.svg)
![JWT](https://img.shields.io/badge/jwt-authentication-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Quantum Safe](https://img.shields.io/badge/quantum--safe-enabled-critical.svg)
![CI](https://img.shields.io/github/actions/workflow/status/thearpankumar/Encrypt5_SecurityProtocols/ci.yml?branch=main)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

---

> **Next-Generation, Quantum-Resistant Cryptographic Toolkit**  
> GPU-accelerated AES, post-quantum Dilithium/Kyber, Double Ratchet secure messaging, Rust P2P networking, JWT authentication, Azure cloud integration, and more.  
> **Optimized for high-throughput and future-proof security applications.**

---

## 🌟 Project Highlights

- **Quantum-Resistant Algorithms**: Integrates NIST post-quantum standards like Dilithium & Kyber for robust digital signatures and key exchange.
- **GPU Acceleration**: Leverages CUDA and Rust for blazing-fast AES and compression suitable for large data and high-performance needs.
- **End-to-End Secure Messaging**: Implements Double Ratchet for modern, forward-secret communication protocols.
- **Multi-Language**: Python (main logic), CUDA (GPU), Rust (networking & low-level crypto), C++, and Jupyter Notebooks for demos.
- **Cloud-Ready**: Azure integration and JWT-based authentication for scalable, secure deployments.
- **P2P Networking**: Rust-powered peer-to-peer stack for decentralized, encrypted communication.
- **API-Ready**: Designed to integrate with FastAPI for rapid development of secure services.
- **Extensive Testing & Notebooks**: Jupyter and JSON utilities for testing, validating, and demoing cryptographic routines.

---

## 🗂️ Repository Structure

| Folder                         | Description                                                                                                   | Technologies       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------ |
| `AesRust/`                     | High-speed AES encryption using Rust and GPU for secure and efficient data protection.                        | Rust, CUDA         |
| `Azure/`                       | Secure integration modules for Microsoft Azure cloud services.                                                 | Python             |
| `Dilithium_ECDH_kyber_signature/` | NIST PQC: Dilithium (signatures), Kyber (key exchange), and ECDH implementations for quantum-safe operations. | Python, C++, Rust  |
| `Double_rachet/`               | Double Ratchet Algorithm for forward-secret, end-to-end encrypted messaging protocols.                        | Python             |
| `Dumy_data_tester_json/`       | Utilities and sample JSON data for encryption and decryption testing.                                          | Python, JSON       |
| `GPU_Encryption/`              | CUDA-powered encryption modules for high-throughput, parallel data processing.                                 | CUDA, Python       |
| `GPU_compression/`             | GPU-accelerated data compression utilities to maximize storage and transmission efficiency.                   | CUDA               |
| `Jwt/`                         | JWT-based authentication and authorization modules.                                                            | Python             |
| `P2P_Rust/`                    | Peer-to-peer networking stack for decentralized, secure communication.                                         | Rust               |

---

## 🧰 Technologies Used

- **Python 3.9+**  
- **Rust 1.73+**
- **CUDA 12.1+**
- **C++17**
- **FastAPI (latest)**
- **Jupyter Notebook**
- **Azure SDKs**
- **JSON Web Token (JWT)**
- **OpenSSL, PyCryptodome, rust-crypto**

---

## ⚙️ Supported Features

- **Hybrid Cryptography**: Combine classical and quantum-safe primitives seamlessly.
- **Performance-Oriented**: Designed for large-scale, latency-sensitive, and throughput-intensive environments.
- **Security Best Practices**: Modern cryptographic standards, forward secrecy, and robust authentication.
- **Cloud & Edge Ready**: Easily integrate with cloud platforms and IoT/edge devices.
- **Continuous Integration**: Automated CI for all pushes and pull requests.

---

## 🚀 Getting Started

1. **Clone the Repository**
   ```sh
   git clone https://github.com/thearpankumar/Encrypt5_SecurityProtocols.git
   cd Encrypt5_SecurityProtocols
   ```

2. **Explore Each Module**
   - Each folder contains a README or code comments for setup and usage.
   - Jupyter Notebooks provide hands-on demonstrations and experiments.

3. **Setup Dependencies**
   - Most modules require Python 3.9+, CUDA-enabled GPU, and the Rust toolchain.
   - See individual module instructions for environment setup.

---

## 💡 Use Cases

- **Quantum-Ready Secure Communications:** Future-proof your messaging and networking.
- **Cloud Security:** Integrate robust cryptography into Azure and other cloud workflows.
- **High-Performance Data Protection:** Encrypt and compress data at speed for big data and IoT.
- **Research & Prototyping:** Use as a foundation for cryptographic research, teaching, or rapid prototyping.

---

## 📚 Documentation

- Each folder contains example scripts and usage notes.
- For advanced usage and API references, see [docs/](docs/) or code comments.

---

## 🤝 Contributing

Contributions are welcome!  
For bug reports, feature requests, or discussions, please open an issue or pull request.

---

## 🛡️ License

This project is licensed under the MIT License.  
See [LICENSE](LICENSE) for full details.

---

## 🙏 Acknowledgements

- NIST PQC Project
- OpenSSL, PyCryptodome, rust-crypto
- Signal Protocol & Double Ratchet paper
- CUDA and NVIDIA for GPU acceleration

---

> **Built with passion by [thearpankumar](https://github.com/thearpankumar) and contributors.**
