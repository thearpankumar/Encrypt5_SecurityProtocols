#include <fstream>
#include <stdio.h>
#include <iostream>
#include <cuda.h>

extern "C" {
    void decrypt_files(const char* encrypted_file, const char* key_file, const char* decrypted_file);
}

long long int read_file_to_memmory(FILE *pInfile, int *pPointer) {
    if(pInfile != NULL) {
        int mIndex = 0;
        int mSize = fread(pPointer + mIndex, 1, sizeof(int), pInfile);
        long long int mFileSize = 0;
        while(mSize != 0) {
            mFileSize = mFileSize + mSize;
            ++mIndex;
            mSize = fread(pPointer + mIndex, 1, mSize, pInfile);
        }
        return mFileSize;
    }
    return 0;
}

long long int write_file_from_memmory(FILE *pOutFile, int *pPointer, long long int pFileSize) {
    if(pOutFile != NULL) {
        pFileSize = fwrite(pPointer, 1, pFileSize, pOutFile);
        return pFileSize;
    }
    return 0;
}

__global__ void generate_decrypted(int *pDataPointer, int *pRandomData, int *pEncryptedData, long long int pSize) {
    long long int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index <= (pSize / sizeof(int))) {
        (*(pEncryptedData + index)) = (*(pDataPointer + index)) ^ (*(pRandomData + index));
    }
}

void decrypt_files(const char* encrypted_file, const char* key_file, const char* decrypted_file) {
    FILE *inFile;
    FILE *outFile;
    FILE *keyFile;
    inFile = fopen(encrypted_file, "rb");
    keyFile = fopen(key_file, "rb");
    outFile = fopen(decrypted_file, "wb");

    if (!inFile) {
        std::cerr << "Error: Could not open encrypted file " << encrypted_file << std::endl;
        return;
    }
    if (!keyFile) {
        std::cerr << "Error: Could not open key file " << key_file << std::endl;
        return;
    }
    if(!outFile) {
        std::cerr << "Error: Could not open output file " << decrypted_file << std::endl;
        return;
    }

    int *encryptedDataPointer = new int[268435456];
    long long int fileSize = read_file_to_memmory(inFile, encryptedDataPointer);
    int *keyDataPointer = new int[fileSize / sizeof(int) + 100];
    int *decryptedDataPointer = new int[fileSize / sizeof(int) + 100];
    fileSize = read_file_to_memmory(keyFile, keyDataPointer);

    int *d_encryptedDataPointer;
    cudaMalloc((void**)&d_encryptedDataPointer, fileSize);
    int *d_keyPointer;
    cudaMalloc((void**)&d_keyPointer, fileSize);
    int *d_decryptedDataPointer;
    cudaMalloc((void**)&d_decryptedDataPointer, fileSize);

    cudaMemcpy(d_encryptedDataPointer, encryptedDataPointer, fileSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyPointer, keyDataPointer, fileSize, cudaMemcpyHostToDevice);

    generate_decrypted<<<fileSize / 64 + 1, 64>>>(d_encryptedDataPointer, d_keyPointer, d_decryptedDataPointer, fileSize);

    cudaMemcpy(decryptedDataPointer, d_decryptedDataPointer, fileSize, cudaMemcpyDeviceToHost);
    fileSize = write_file_from_memmory(outFile, decryptedDataPointer, fileSize);

    fclose(inFile);
    fclose(outFile);
    fclose(keyFile);
}