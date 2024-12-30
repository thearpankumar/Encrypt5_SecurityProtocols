#include <fstream>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include "random.h" // Assuming this contains the necessary headers for rand()
long long int  read_file_to_memmory(FILE *pInfile , int *pPointer)
{
    if(pInfile != NULL)
    {
        
        int mIndex =0;
        int mSize = fread(pPointer+mIndex,1,sizeof(int),pInfile);
        long long int mFileSize=0;
        while(mSize!= 0)
        {
            mFileSize = mFileSize +mSize;
            ++mIndex;
            mSize = fread(pPointer+mIndex,1,mSize,pInfile);
        }
        return mFileSize;
    }
    return 0;
}
long long int write_file_from_memmory(FILE *pOutFile , int *pPointer,long long int pFileSize)
{
    if(pOutFile!=NULL)
    {
        pFileSize = fwrite(pPointer,1,pFileSize,pOutFile);
        return pFileSize;
    }
    return 0;
}
long long int generate_random_bits(int  *pPointer , long long int pSize)
{
    long long int mSize = pSize;
    long long int mIndex =0;
    while(pSize>0)
    {
        (*(pPointer+mIndex)) = rand();
        ++mIndex;
        pSize = pSize - sizeof(int);
    }
    return mSize;
}
__global__ void generate_encrypted(int *pDataPointer , int *pRandomData , int *pEncryptedData , long long int pSize)
{
    long long int index = blockIdx.x * blockDim.x + threadIdx.x;
    if( index <=(pSize /sizeof(int) ))
    {
        (*(pEncryptedData+index)) = (*(pDataPointer+ index))^(*(pRandomData+index));
    }
    else
        return;
}
int main(int argc , char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <encrypted_file> <key_file>" << std::endl;
        return 1;
    }
    FILE *inFile;
    FILE *outFile;
    FILE *keyFile;
    inFile = fopen(argv[1],"rb");
    outFile = fopen(argv[2],"wb");
     keyFile = fopen(argv[3],"wb");

     if (!inFile)
    {
        std::cerr << "Error: Could not open input file " << argv[1] << std::endl;
        return 1;
    }
    if(!outFile)
    {
        std::cerr << "Error: Could not open encrypted file " << argv[2] << std::endl;
        return 1;
    }
     if (!keyFile)
    {
        std::cerr << "Error: Could not open key file " << argv[3] << std::endl;
        return 1;
    }

    int *dataPointer = new int[268435456];
    long long int fileSize = read_file_to_memmory(inFile,dataPointer);
    int *randomBytePointer = new int[fileSize/sizeof(int) + 100];
    fileSize = generate_random_bits(randomBytePointer , fileSize);
    int *encryptedPointer = new int[fileSize/sizeof(int) +100];
    int *d_dataPointer;
    int *d_randomBytePointer;
    int *d_EncryptedData;
    cudaMalloc((void**)&d_dataPointer,fileSize);
    cudaMalloc((void**)&d_randomBytePointer,fileSize);
    cudaMalloc((void**)&d_EncryptedData ,fileSize);
    cudaMemcpy(d_dataPointer,dataPointer,fileSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_randomBytePointer,randomBytePointer,fileSize,cudaMemcpyHostToDevice);
    generate_encrypted<<<fileSize/64 + 1,64>>>(d_dataPointer,d_randomBytePointer,d_EncryptedData,fileSize);
    cudaMemcpy(encryptedPointer,d_EncryptedData,fileSize,cudaMemcpyDeviceToHost);
    fileSize =write_file_from_memmory(outFile,encryptedPointer,fileSize);
    fileSize =write_file_from_memmory(keyFile,randomBytePointer,fileSize);
    fclose(inFile);
    fclose(outFile);
    fclose(keyFile);
}