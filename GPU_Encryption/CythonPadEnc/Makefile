# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -shared -Xcompiler -fPIC
BUILD_DIR = build/
# Target shared libraries
BUILD_DIR = build
ENCRYPT_LIB = $(BUILD_DIR)/libaesencrypt.so
DECRYPT_LIB = $(BUILD_DIR)/libaesdecrypt.so
PAD_ENCRYPT_LIB = $(BUILD_DIR)/padlibencrypt.so
PAD_DECRYPT_LIB = $(BUILD_DIR)/padlibdecrypt.so

# Source files
ENCRYPT_SRC = Aesencrypt.cu
DECRYPT_SRC = Aesdecrypt.cu
PAD_ENCRYPT_SRC = encrypt.cu
PAD_DECRYPT_SRC = decrypt.cu

# Files to clean
CLEAN_FILES = $(BUILD_DIR) key.txt encrypted.bin decrypted.txt

# Default target
all: $(ENCRYPT_LIB) $(DECRYPT_LIB) $(PAD_ENCRYPT_LIB) $(PAD_DECRYPT_LIB)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build encryption library
$(ENCRYPT_LIB): $(ENCRYPT_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build decryption library
$(DECRYPT_LIB): $(DECRYPT_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build padding encryption library
$(PAD_ENCRYPT_LIB): $(PAD_ENCRYPT_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build padding decryption library
$(PAD_DECRYPT_LIB): $(PAD_DECRYPT_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Clean up generated files and build directory
clean:
	rm -rf $(CLEAN_FILES)

# Phony targets
.PHONY: all clean