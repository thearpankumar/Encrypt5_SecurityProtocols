// This is the final version of Server 1.
// It accepts the IP, HTTP port, and P2P port of Server 2 as command-line arguments.

use futures::StreamExt;
use libp2p::{
    gossipsub,
    identity,
    kad,
    noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp,
    yamux,
    Multiaddr,
    PeerId,
    SwarmBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    // Import `env` to read command-line arguments
    env,
    collections::HashMap,
    error::Error,
    fmt,
    fs,
    io::{stdout, Write},
    path::Path,
    sync::Arc,
};
use tokio::{io::AsyncBufReadExt, sync::Mutex, time::Duration};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use sha2::{Digest, Sha256};
use bs58;
use reqwest;


// --- All your existing structs and helper functions are the same ---
#[derive(Debug)]
enum CryptoError { AesGcm(aes_gcm::Error), Utf8(std::string::FromUtf8Error), Base58(bs58::decode::Error), PeerId(libp2p::identity::ParseError) }
impl fmt::Display for CryptoError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { match self { CryptoError::AesGcm(e) => write!(f, "AES-GCM error: {:?}", e), CryptoError::Utf8(e) => write!(f, "UTF-8 error: {}", e), CryptoError::Base58(e) => write!(f, "Base58 decoding error: {}", e), CryptoError::PeerId(e) => write!(f, "PeerId parsing error: {}", e), } } }
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


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    // --- NEW: Read Server 2's IP and ports from command-line arguments ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("🔥 Error: Missing arguments.");
        eprintln!("Usage: cargo run -- <SERVER_2_IP> <HTTP_PORT> <P2P_PORT>");
        eprintln!("Example: cargo run -- 127.0.0.1 8080 4001");
        return Err("Missing arguments".into());
    }

    let server2_ip = &args[1];
    let server2_http_port: u16 = args[2].parse().map_err(|_| "Invalid HTTP port number")?;
    let server2_p2p_port: u16 = args[3].parse().map_err(|_| "Invalid P2P port number")?;

    // --- Libp2p Setup (unchanged) ---
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("✅ Server 1 (General Func) dynamic PeerId: {local_peer_id}");

    let key_path = Path::new("swarm.key");
    let _key = get_or_create_key(key_path)?;

    let mut swarm = SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key| {
            let gossipsub_config = gossipsub::ConfigBuilder::default().build().unwrap();
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            ).unwrap();
            let kademlia =
                kad::Behaviour::new(local_peer_id, kad::store::MemoryStore::new(local_peer_id));
            Ok(ChatBehaviour { gossipsub, kademlia })
        })?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    let chat_topic = gossipsub::IdentTopic::new("private-chat-room");
    swarm.behaviour_mut().gossipsub.subscribe(&chat_topic)?;
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;


    // --- Dynamic PeerId Discovery using the provided IP and ports ---
    let http_url = format!("http://{}:{}/peer_id", server2_ip, server2_http_port);
    println!("🔎 Fetching PeerId from Server 2 at {}...", http_url);

    match reqwest::get(&http_url).await {
        Ok(response) => match response.json::<String>().await {
            Ok(remote_peer_id_str) => {
                println!("✅ Got PeerId: {}", remote_peer_id_str);
                let server2_multiaddr_str = format!(
                    "/ip4/{}/tcp/{}/p2p/{}",
                    server2_ip, server2_p2p_port, remote_peer_id_str
                );
                match server2_multiaddr_str.parse::<Multiaddr>() {
                    Ok(remote_addr) => {
                        swarm.dial(remote_addr.clone())?;
                        println!("📞 Dialing master key server at {}", server2_multiaddr_str);
                        if let Some(libp2p::multiaddr::Protocol::P2p(peer_id)) = remote_addr.iter().last() {
                            swarm.behaviour_mut().kademlia.add_address(&peer_id, remote_addr);
                            swarm.behaviour_mut().kademlia.bootstrap()?;
                        }
                    }
                    Err(e) => eprintln!("🔥 Failed to parse the constructed multiaddress: {}", e),
                }
            }
            Err(e) => eprintln!("🔥 Failed to parse JSON response from Server 2: {}.", e),
        },
        Err(e) => eprintln!("🔥 Failed to connect to Server 2 at {}: {}", http_url, e),
    }

    // --- Event Loop (unchanged) ---
    let connected_peers = Arc::new(Mutex::new(HashMap::<PeerId, bool>::new()));
    let nonce_counters = Arc::new(Mutex::new(HashMap::<PeerId, u64>::new()));
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();

    loop {
        tokio::select! {
            line = stdin.next_line() => {
                if let Some(line) = line?.as_deref() {
                     if !line.trim().is_empty() {
                        let peers = connected_peers.lock().await;
                        if peers.is_empty() {
                            println!("⚠️ No peers connected. Cannot send message.");
                        } else {
                            for &peer_id in peers.keys() {
                                let session_key = derive_session_key(&local_peer_id, &peer_id);
                                let cipher = Aes256Gcm::new_from_slice(&session_key)?;
                                let mut nonces = nonce_counters.lock().await;
                                let nonce_counter = nonces.entry(peer_id).or_insert(0);
                                *nonce_counter += 1;
                                let mut nonce = [0u8; 12];
                                nonce[4..12].copy_from_slice(&nonce_counter.to_be_bytes());
                                let encrypted_content = encrypt_message(&cipher, line.trim(), &nonce)?;
                                let chat_msg = ChatMessage::Chat { sender: local_peer_id.to_string(), content: encrypted_content, nonce: nonce.to_vec() };
                                let json = serde_json::to_string(&chat_msg)?;
                                if let Err(e) = swarm.behaviour_mut().gossipsub.publish(chat_topic.clone(), json.as_bytes()) {
                                     eprintln!("Error publishing message: {:?}", e);
                                }
                            }
                        }
                    }
                }
                print!("> ");
                stdout().flush()?;
            }
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("👂 Listening on {}", address.with_p2p(local_peer_id).unwrap());
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        println!("🤝 Connection established with {peer_id}");
                        connected_peers.lock().await.insert(peer_id, true);
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
                    SwarmEvent::Behaviour(_) => {},
                    SwarmEvent::ConnectionClosed { peer_id, .. } => {
                        println!("🔌 Connection lost with {peer_id}");
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