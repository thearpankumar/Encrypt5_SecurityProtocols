#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <vector>
#define BYTE unsigned char

using namespace std;

class aes_block
{
public:
    BYTE block[16];
};

void printBytes(BYTE b[], int len) {
    int i;
    for (i=0; i<len; i++)
        printf("%x ", b[i]);
    printf("\n");
}

void f1printBytes(BYTE b[], int len, FILE* fp) {
    int i;
    for (i=0; i<len; i++)
        fprintf(fp, "%02x ", b[i]);
    fprintf(fp, "\n");
}

int flag=0;
void f2printBytes(BYTE b[], int len, FILE* fp) {
    int i;
    for (i=0; i<len; i++){
        fprintf(fp, "%c", b[i]);
        if(b[i]=='\n')
            flag++;
    }
}

void f3printBytes(BYTE b[], int len, FILE* fp) {
    int i;
    for (i=0; i<len; i++){
        if(b[i]=='\0'){
            return ;
        }
        fprintf(fp, "%c", b[i]);
        if(b[i]=='\n')
            flag++;
    }
}

/******************************************************************************/
// The following lookup tables and functions are for internal use only!
// Host Side Tables

BYTE AES_Sbox_host[] =
{
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

//Device Side LUTs
__constant__ BYTE AES_Sbox[] =
{
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant__ BYTE AES_ShiftRowTab[16] = {0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11};
__constant__ BYTE AES_Sbox_Inv[256];
__constant__ BYTE AES_ShiftRowTab_Inv[16];
__constant__ BYTE AES_xtime[256];

__device__ inline BYTE subBytes(BYTE in, const BYTE* sbox) {
    return sbox[in];
}
__device__ inline void AES_SubBytes(BYTE state[], const BYTE* sbox) {
    state[0] = subBytes(state[0], sbox); state[1] = subBytes(state[1], sbox); state[2] = subBytes(state[2], sbox); state[3] = subBytes(state[3], sbox);
    state[4] = subBytes(state[4], sbox); state[5] = subBytes(state[5], sbox); state[6] = subBytes(state[6], sbox); state[7] = subBytes(state[7], sbox);
    state[8] = subBytes(state[8], sbox); state[9] = subBytes(state[9], sbox); state[10] = subBytes(state[10], sbox); state[11] = subBytes(state[11], sbox);
    state[12] = subBytes(state[12], sbox); state[13] = subBytes(state[13], sbox); state[14] = subBytes(state[14], sbox); state[15] = subBytes(state[15], sbox);

}

__device__ inline void AES_AddRoundKey(BYTE state[], BYTE rkey[]) {
    for(int i = 0; i < 16; i++)
        state[i] ^= rkey[i];
}


__device__ inline void AES_ShiftRows(BYTE state[], const BYTE* shifttab) {
     BYTE h[16];
    memcpy(h, state, 16);
    state[0] = h[shifttab[0]]; state[1] = h[shifttab[1]]; state[2] = h[shifttab[2]]; state[3] = h[shifttab[3]];
    state[4] = h[shifttab[4]]; state[5] = h[shifttab[5]]; state[6] = h[shifttab[6]]; state[7] = h[shifttab[7]];
    state[8] = h[shifttab[8]]; state[9] = h[shifttab[9]]; state[10] = h[shifttab[10]]; state[11] = h[shifttab[11]];
    state[12] = h[shifttab[12]]; state[13] = h[shifttab[13]]; state[14] = h[shifttab[14]]; state[15] = h[shifttab[15]];
}



__device__ inline void AES_MixColumns(BYTE state[], const BYTE* xtime) {
   BYTE s0, s1, s2, s3, h;
    for (int i = 0; i < 16; i += 4) {
        s0 = state[i + 0]; s1 = state[i + 1]; s2 = state[i + 2]; s3 = state[i + 3];
        h = s0 ^ s1 ^ s2 ^ s3;
        state[i + 0] ^= h ^ xtime[s0 ^ s1];
        state[i + 1] ^= h ^ xtime[s1 ^ s2];
        state[i + 2] ^= h ^ xtime[s2 ^ s3];
        state[i + 3] ^= h ^ xtime[s3 ^ s0];
    }
}


__device__ inline void AES_MixColumns_Inv(BYTE state[], const BYTE* xtime) {
    BYTE s0, s1, s2, s3, h, xh, h1, h2;
    for (int i = 0; i < 16; i += 4) {
        s0 = state[i + 0]; s1 = state[i + 1]; s2 = state[i + 2]; s3 = state[i + 3];
        h = s0 ^ s1 ^ s2 ^ s3;
        xh = xtime[h];
        h1 = xtime[xtime[xh ^ s0 ^ s2]] ^ h;
        h2 = xtime[xtime[xh ^ s1 ^ s3]] ^ h;
        state[i + 0] ^= h1 ^ xtime[s0 ^ s1];
        state[i + 1] ^= h2 ^ xtime[s1 ^ s2];
        state[i + 2] ^= h1 ^ xtime[s2 ^ s3];
        state[i + 3] ^= h2 ^ xtime[s3 ^ s0];
    }
}

__device__ void AES_Init() {
    int i;
    for(i = 0; i < 256; i++){
        AES_Sbox_Inv[AES_Sbox[i]] = i;
    }
    for(i = 0; i < 16; i++)
        AES_ShiftRowTab_Inv[AES_ShiftRowTab[i]] = i;
    for(i = 0; i < 128; i++) {
        AES_xtime[i] = i << 1;
        AES_xtime[128 + i] = (i << 1) ^ 0x1b;
    }
}


// AES_ExpandKey: expand a cipher key. Depending on the desired encryption 
// strength of 128, 192 or 256 bits 'key' has to be a byte array of length 
// 16, 24 or 32, respectively. The key expansion is done "in place", meaning 
// that the array 'key' is modified.
int AES_ExpandKey(BYTE key[], int keyLen) {
    int kl = keyLen, ks, Rcon = 1, i, j;
    BYTE temp[4], temp2[4];
    switch (kl) {
        case 16: ks = 16 * (10 + 1); break;
        case 24: ks = 16 * (12 + 1); break;
        case 32: ks = 16 * (14 + 1); break;
        default: 
        printf("AES_ExpandKey: Only key lengths of 16, 24 or 32 bytes allowed!");
    }
    for(i = kl; i < ks; i += 4) {
        memcpy(temp, &key[i-4], 4);
        if (i % kl == 0) {
            temp2[0] = AES_Sbox_host[temp[1]] ^ Rcon;
            temp2[1] = AES_Sbox_host[temp[2]];
            temp2[2] = AES_Sbox_host[temp[3]];
            temp2[3] = AES_Sbox_host[temp[0]];
            memcpy(temp, temp2, 4);
            if ((Rcon <<= 1) >= 256)
                Rcon ^= 0x11b;
        }
        else if ((kl > 24) && (i % kl == 16)) {
            temp2[0] = AES_Sbox_host[temp[0]];
            temp2[1] = AES_Sbox_host[temp[1]];
            temp2[2] = AES_Sbox_host[temp[2]];
            temp2[3] = AES_Sbox_host[temp[3]];
            memcpy(temp, temp2, 4);
        }
        for(j = 0; j < 4; j++)
            key[i + j] = key[i + j - kl] ^ temp[j];
    }
    return ks;
}
__global__ void initialize_device(){
    AES_Init();
}
// AES_Encrypt: encrypt the 16 byte array 'block' with the previously expanded key 'key'.
__global__ void AES_Encrypt(aes_block aes_block_array[], BYTE key[], int keyLen, int block_number) {
    int global_thread_index = blockDim.x*blockIdx.x + threadIdx.x;
        
    if(global_thread_index < block_number){
        BYTE block[16];
        for(int i=0; i<16; i++)
            block[i] = aes_block_array[global_thread_index].block[i];
        int l = keyLen, i;
        AES_AddRoundKey(block, &key[0]);
         #pragma unroll
        for(i = 16; i < l - 16; i += 16) {
            AES_SubBytes(block, AES_Sbox);
            AES_ShiftRows(block, AES_ShiftRowTab);
            AES_MixColumns(block, AES_xtime);
            AES_AddRoundKey(block, &key[i]);
        }
        AES_SubBytes(block, AES_Sbox);
        AES_ShiftRows(block, AES_ShiftRowTab);
        AES_AddRoundKey(block, &key[i]);
        for(int i=0; i<16; i++)
           aes_block_array[global_thread_index].block[i] = block[i];
    }
}

// AES_Decrypt: decrypt the 16 byte array 'block' with the previously expanded key 'key'.

__global__ void AES_Decrypt(aes_block aes_block_array[], BYTE key[], int keyLen, int block_number) {
    int global_thread_index = blockDim.x*blockIdx.x + threadIdx.x;

    if(global_thread_index < block_number){
        BYTE block[16];
        for(int i=0; i<16; i++)
            block[i] = aes_block_array[global_thread_index].block[i];
        int l = keyLen, i;
        AES_AddRoundKey(block, &key[l - 16]);
        AES_ShiftRows(block, AES_ShiftRowTab_Inv);
         #pragma unroll
         for(int i=0; i<16; i++) {
          block[i] = AES_Sbox_Inv[block[i]];
        }

        for(i = l - 32; i >= 16; i -= 16) {
            AES_AddRoundKey(block, &key[i]);
            AES_MixColumns_Inv(block, AES_xtime);
            AES_ShiftRows(block, AES_ShiftRowTab_Inv);
             #pragma unroll
           for(int i=0; i<16; i++) {
             block[i] = AES_Sbox_Inv[block[i]];
           }
        }
        AES_AddRoundKey(block, &key[0]);
         for(int i=0; i<16; i++)
            aes_block_array[global_thread_index].block[i] = block[i];
    }
}

void handleCudaError(cudaError_t error, const char *file, int line)
{
	if (error != cudaSuccess) {
		printf("CUDA Error %s:%d: %s\n", file, line, cudaGetErrorString(error));
		exit(1);
	}
}

// ===================== test ============================================
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input_file> <key_file> <encrypted_file> <decrypted_file>" << endl;
        return 1;
    }
    ifstream ifs;
    ifs.open(argv[1], std::ifstream::binary);
    if(!ifs){
        cerr<<"Cannot open the input file"<<endl;
        exit(1);
    }
    ifs.seekg(0, ios::end);
    int infileLength = ifs.tellg();
    ifs.seekg (0, ios::beg);
    cout<<"Length of input file: "<<infileLength<<endl;

    int block_number = (infileLength + 15) / 16; // Ceiling division
  //  int number_of_zero_pending = (block_number*16)- infileLength;
    vector<aes_block> aes_block_vector(block_number);

    BYTE key[16 * (14 + 1)];
    int keyLen = 0;

    ifstream key_fp;
    key_fp.open(argv[2]);
    while(key_fp.peek()!=EOF)
    {
            key_fp>>key[keyLen];
            if(key_fp.eof())
                break;
            keyLen++;
    }

    cout<<"Key length: "<<keyLen<<endl;
    switch (keyLen)
    {
    case 16:break;
    case 24:break;
    case 32:break;
    default:printf("ERROR : keyLen should be 128, 192, 256bits\n"); return 0;
    }


    int expandKeyLen = AES_ExpandKey(key, keyLen);

    char temp[16];
    for(int i=0; i<block_number; i++){
        if(i*16 < infileLength){
             ifs.read(temp, 16);
            for(int j=0; j<16; j++){
                if(i*16 + j < infileLength){
                      aes_block_vector[i].block[j] = (unsigned char)temp[j];
                }
                else
                    aes_block_vector[i].block[j] = 0;
                
            }
        }
        else {
            for(int j=0; j<16; j++){
                aes_block_vector[i].block[j] = 0;
            }
        }
       
    }
    ifs.close();
	cudaSetDevice(0);	//device 0: Tesla K20c, device 1: GTX 770, device 1 is faster for this application
	cudaDeviceProp prop;
    cudaError_t cuda_err;
	cuda_err = cudaGetDeviceProperties(&prop, 0);
    handleCudaError(cuda_err, __FILE__, __LINE__);
	int num_sm = prop.multiProcessorCount;

    aes_block *cuda_aes_block_array;
    BYTE *cuda_key;

    int thrdperblock = block_number/num_sm;
    if(block_number%num_sm>0)
        thrdperblock++;

    if(thrdperblock>1024){
        thrdperblock = 1024;
        num_sm = block_number/1024;
        if(block_number%1024>0){
            num_sm++;
        }
    }

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);
    printf("Number of SMs: %d\nThreads per block: %d\nBlocks per Grid: %d\n", num_sm, thrdperblock, num_sm);

	
	cuda_err = cudaMalloc(&cuda_aes_block_array, block_number*sizeof(aes_block));
	handleCudaError(cuda_err, __FILE__, __LINE__);
	cuda_err = cudaMalloc(&cuda_key,16*15*sizeof(BYTE) );
	handleCudaError(cuda_err, __FILE__, __LINE__);
	
    cuda_err = cudaMemcpy(cuda_aes_block_array, aes_block_vector.data(), block_number*sizeof(aes_block), cudaMemcpyHostToDevice);
    handleCudaError(cuda_err, __FILE__, __LINE__);
    cuda_err = cudaMemcpy(cuda_key, key, 16*15*sizeof(BYTE), cudaMemcpyHostToDevice);
    handleCudaError(cuda_err, __FILE__, __LINE__);
	cuda_err = cudaMemcpyToSymbol(AES_Sbox_Inv, AES_Sbox_host, 256*sizeof(BYTE));
	handleCudaError(cuda_err, __FILE__, __LINE__);
    
    initialize_device<<<1,1>>>();
    cuda_err = cudaGetLastError();
    handleCudaError(cuda_err, __FILE__, __LINE__);
     
    AES_Encrypt <<< BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key, expandKeyLen, block_number);
     cuda_err = cudaGetLastError();
    handleCudaError(cuda_err, __FILE__, __LINE__);
    cuda_err = cudaMemcpy(aes_block_vector.data(), cuda_aes_block_array, block_number*sizeof(aes_block), cudaMemcpyDeviceToHost);
     handleCudaError(cuda_err, __FILE__, __LINE__);

     FILE* en_fp = fopen(argv[3], "wb");
    for(int i=0; i<block_number; i++){
            f1printBytes(aes_block_vector[i].block, 16, en_fp);
    }
    fclose(en_fp);
	
    AES_Decrypt <<< BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key, expandKeyLen, block_number);
     cuda_err = cudaGetLastError();
    handleCudaError(cuda_err, __FILE__, __LINE__);

     cuda_err = cudaMemcpy(aes_block_vector.data(), cuda_aes_block_array, block_number*sizeof(aes_block), cudaMemcpyDeviceToHost);
    handleCudaError(cuda_err, __FILE__, __LINE__);
     FILE* de_fp = fopen(argv[4], "wb");
    for(int i=0; i<block_number-1; i++){
        f2printBytes(aes_block_vector[i].block, 16, de_fp);
    }
    f3printBytes(aes_block_vector[block_number-1].block, 16, de_fp);
    
    fclose(de_fp);

	cuda_err = cudaFree(cuda_aes_block_array);
	handleCudaError(cuda_err, __FILE__, __LINE__);
	cuda_err = cudaFree(cuda_key);
	handleCudaError(cuda_err, __FILE__, __LINE__);
    return 0;
}