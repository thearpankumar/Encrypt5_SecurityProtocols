// This is the final, corrected version for Server 2.
// Fixes the Arc<String> Serialize error.

use futures::StreamExt;
use libp2p::{
    gossipsub,
    identity,
    kad,
    noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp,
    yamux,
    PeerId,
    SwarmBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    fs,
    io::{stdout, Write},
    path::Path,
    sync::Arc,
    net::SocketAddr,
};
use tokio::{io::AsyncBufReadExt, sync::Mutex, time::Duration};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use sha2::{Digest, Sha256};
use bs58;

// Imports for the Axum web server
use axum::{routing::get, Json, Router, Extension};


// --- Structs and helper functions ---
#[derive(Debug)]
enum CryptoError { AesGcm(aes_gcm::Error), Utf8(std::string::FromUtf8Error), Base58(bs58::decode::Error), PeerId(libp2p::identity::ParseError) }
impl fmt::Display for CryptoError { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result { match self { CryptoError::AesGcm(e) => write!(f, "AES-GCM error: {:?}", e), CryptoError::Utf8(e) => write!(f, "UTF-8 error: {}", e), CryptoError::Base58(e) => write!(f, "Base58 decoding error: {}", e), CryptoError::PeerId(e) => write!(f, "PeerId parsing error: {}", e), } } }
impl std::error::Error for CryptoError {}
impl From<aes_gcm::Error> for CryptoError { fn from(err: aes_gcm::Error) -> Self { CryptoError::AesGcm(err) } }
impl From<std::string::FromUtf8Error> for CryptoError { fn from(err: std::string::FromUtf8Error) -> Self { CryptoError::Utf8(err) } }
impl From<bs58::decode::Error> for CryptoError { fn from(err: bs58::decode::Error) -> Self { CryptoError::Base58(err) } }
impl From<libp2p::identity::ParseError> for CryptoError { fn from(err: libp2p::identity::ParseError) -> Self { CryptoError::PeerId(err) } }

#[derive(NetworkBehaviour)]
#[behaviour(out_event = "ChatBehaviourEvent")]
struct ChatBehaviour { gossipsub: gossipsub::Behaviour, kademlia: kad::Behaviour<kad::store::MemoryStore> }
#[derive(Debug)]
enum ChatBehaviourEvent { Gossipsub(gossipsub::Event), Kademlia(kad::Event) }
impl From<gossipsub::Event> for ChatBehaviourEvent { fn from(event: gossipsub::Event) -> Self { ChatBehaviourEvent::Gossipsub(event) } }
impl From<kad::Event> for ChatBehaviourEvent { fn from(event: kad::Event) -> Self { ChatBehaviourEvent::Kademlia(event) } }

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum ChatMessage { Chat { sender: String, content: Vec<u8>, nonce: Vec<u8> } }
fn derive_session_key(sender: &PeerId, receiver: &PeerId) -> [u8; 32] { let mut hasher = Sha256::new(); let (first, second) = if sender < receiver { (sender, receiver) } else { (receiver, sender) }; hasher.update(first.to_bytes()); hasher.update(second.to_bytes()); let result = hasher.finalize(); let mut key = [0u8; 32]; key.copy_from_slice(&result[..32]); key }
fn encrypt_message(cipher: &Aes256Gcm, plaintext: &str, nonce: &[u8; 12]) -> Result<Vec<u8>, CryptoError> { Ok(cipher.encrypt(Nonce::from_slice(nonce), plaintext.as_bytes())?) }
fn decrypt_message(cipher: &Aes256Gcm, ciphertext: &[u8], nonce: &[u8]) -> Result<String, CryptoError> { let plaintext = cipher.decrypt(Nonce::from_slice(nonce), ciphertext)?; Ok(String::from_utf8(plaintext)?) }
fn get_or_create_key(path: &Path) -> Result<[u8; 32], Box<dyn Error>> { if path.exists() { let key_bytes = fs::read(path)?; Ok(key_bytes.try_into().map_err(|_| "Swarm key file must be exactly 32 bytes.")?) } else { let key: [u8; 32] = rand::random(); fs::write(path, key)?; Ok(key) } }


// *** THE FIX IS HERE ***
#[axum::debug_handler]
async fn peer_id_handler(
    // We receive the Arc<String> via the Extension
    Extension(peer_id_arc): Extension<Arc<String>>,
    // FIX: Change return type to Json<String> because String implements Serialize
) -> Json<String> {
    // FIX: Clone the String content out of the Arc, and wrap that.
    Json(peer_id_arc.to_string())
}
// ***********************


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    const KEYPAIR_FILENAME: &str = "server2_keypair.key";
    // Added unwrap_or_else for better error message if key file missing
    let key_bytes = fs::read(KEYPAIR_FILENAME).unwrap_or_else(|_| panic!("Failed to read key file: {}. Run key generator first.", KEYPAIR_FILENAME));
    let local_key = identity::Keypair::from_protobuf_encoding(&key_bytes)?;
    let local_peer_id = PeerId::from(local_key.public());
    println!("✅ Server 2 static PeerId: {local_peer_id}");

    let shared_peer_id = Arc::new(local_peer_id.to_string());

    tokio::spawn(async move {
        let app = Router::new()
            .route("/peer_id", get(peer_id_handler))
            .layer(Extension(shared_peer_id)); // Pass the Arc here
        let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
        println!("📡 HTTP server hosting PeerId at http://{}/peer_id", addr);
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    let key_path = Path::new("swarm.key");
    let _key = get_or_create_key(key_path)?;
    let mut swarm = SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key| {
            let gossipsub_config = gossipsub::ConfigBuilder::default().build().unwrap();
            let gossipsub = gossipsub::Behaviour::new(gossipsub::MessageAuthenticity::Signed(key.clone()), gossipsub_config).unwrap();
            let mut kademlia = kad::Behaviour::new(local_peer_id, kad::store::MemoryStore::new(local_peer_id));
            kademlia.set_mode(Some(kad::Mode::Server));
            Ok(ChatBehaviour { gossipsub, kademlia })
        })?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    let chat_topic = gossipsub::IdentTopic::new("private-chat-room");
    swarm.behaviour_mut().gossipsub.subscribe(&chat_topic)?;
    swarm.listen_on("/ip4/0.0.0.0/tcp/4001".parse()?)?;

    let connected_peers = Arc::new(Mutex::new(HashMap::<PeerId, bool>::new()));
    let nonce_counters = Arc::new(Mutex::new(HashMap::<PeerId, u64>::new()));
    let known_peers = Arc::new(Mutex::new(HashSet::<PeerId>::new()));

    println!("👑 Master Key Server p2p node is running. Waiting for connections...");
    let mut _stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();

    loop {
        tokio::select! {
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => println!("👂 P2P Listening on {}", address.with_p2p(local_peer_id).unwrap()),
                    SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                        println!("🤝 P2P Connection established with {peer_id}");
                        swarm.behaviour_mut().kademlia.add_address(&peer_id, endpoint.get_remote_address().clone());
                        connected_peers.lock().await.insert(peer_id, true);
                        known_peers.lock().await.insert(peer_id);
                    }
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Gossipsub(gossipsub::Event::Message { message, .. })) => {
                        if let Ok(chat_msg) = serde_json::from_slice::<ChatMessage>(&message.data) {
                            let ChatMessage::Chat { sender, content, nonce } = chat_msg;
                             if let Ok(sender_bytes) = bs58::decode(&sender).into_vec() {
                                if let Ok(sender_peer_id) = PeerId::from_bytes(&sender_bytes) {
                                     let session_key = derive_session_key(&local_peer_id, &sender_peer_id);
                                     if let Ok(cipher) = Aes256Gcm::new_from_slice(&session_key) {
                                        match decrypt_message(&cipher, &content, &nonce) {
                                            Ok(decrypted) => println!("\r📨 {}: {}", sender, decrypted),
                                            Err(e) => eprintln!("\rDecryption failed from {}: {:?}", sender, e),
                                        }
                                     }
                                }
                             }
                        }
                    }
                     SwarmEvent::Behaviour(_) => {}, // Ignore other behaviour events
                    SwarmEvent::ConnectionClosed { peer_id, .. } => {
                        println!("🔌 P2P Connection lost with {peer_id}");
                        connected_peers.lock().await.remove(&peer_id);
                        nonce_counters.lock().await.remove(&peer_id);
                    }
                    _ => {}
                }
                print!("\r> ");
                stdout().flush()?;
            }
        }
    }
}