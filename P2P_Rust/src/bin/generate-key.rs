// A small utility to save the key
use libp2p::identity;
use std::fs;

fn main() -> std::io::Result<()> {
    let keypair = identity::Keypair::generate_ed25519();
    fs::write("server2_keypair.key", keypair.to_protobuf_encoding().unwrap())?;
    println!("✅ Server 2 key saved to server2_keypair.key");
    let peer_id = libp2p::PeerId::from(keypair.public());
    println!("🔑 Server 2 PeerId: {}", peer_id);
    Ok(())
}