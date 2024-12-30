use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write, Read};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rand::Rng;

const BUFFER_LEN: usize = 128 * 1024 * 1024;
const NONCE_LEN: usize = 12;

fn encrypt_large_file_parallel(
    source_file_path: &str,
    dist_file_path: &str,
    key: &Key<Aes256Gcm>,
) -> Result<(), anyhow::Error> {
    let cipher = Arc::new(Aes256Gcm::new(key));

    let mut source_file = File::open(source_file_path)?;
    let file_size = source_file.metadata()?.len() as usize;

    // Wrap the File with Mutex and Arc to synchronize writes
    let dist_file =  Arc::new(Mutex::new(BufWriter::new(File::create(dist_file_path)?)));

    // Create a shared pre-allocated buffer for the chunks
    let mut buffer = vec![0u8; file_size];
    source_file.read_exact(&mut buffer)?;

    let chunks: Vec<&[u8]> = buffer.chunks(BUFFER_LEN).collect();

    let result: Result<(), anyhow::Error> = chunks.into_par_iter().try_for_each(|chunk| {
        // Generate a unique nonce for this chunk
        let mut rng = rand::thread_rng();
        let mut nonce_bytes = [0u8; NONCE_LEN];
        rng.fill(&mut nonce_bytes[..]);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt the chunk
        let ciphertext = cipher.encrypt(nonce, chunk).map_err(|err| anyhow::anyhow!("Encryption failed: {}", err))?;
    
        let mut temp = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        temp.extend_from_slice(&nonce_bytes);
        temp.extend_from_slice(&ciphertext);
    
        let mut guard = dist_file.lock().map_err(|err| anyhow::anyhow!("Failed to lock destination file: {}", err))?;
        guard.write_all(&temp).map_err(|err| anyhow::anyhow!("Failed to write to destination file: {}", err))?;
    
        Ok(())
    });

    result?;

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    let start_time = Instant::now();
    let key = Aes256Gcm::generate_key(OsRng);
    encrypt_large_file_parallel("large_file.txt", "output.enc", &key)?;
    println!("File encrypted successfully!");
    let elapsed_time = start_time.elapsed();
    println!("Time taken: {} microseconds", elapsed_time.as_micros());
    Ok(())
}