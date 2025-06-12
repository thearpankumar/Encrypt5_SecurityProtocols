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
    collections::{HashMap, HashSet},
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
use rand::Rng;
use sha2::{Digest, Sha256};
use bs58; // For Base58 decoding

#[derive(Debug)]
enum CryptoError {
    AesGcm(aes_gcm::Error),
    Utf8(std::string::FromUtf8Error),
    Base58(bs58::decode::Error),
    PeerId(libp2p::identity::ParseError),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::AesGcm(e) => write!(f, "AES-GCM error: {:?}", e),
            CryptoError::Utf8(e) => write!(f, "UTF-8 error: {}", e),
            CryptoError::Base58(e) => write!(f, "Base58 decoding error: {}", e),
            CryptoError::PeerId(e) => write!(f, "PeerId parsing error: {}", e),
        }
    }
}

impl std::error::Error for CryptoError {}

impl From<aes_gcm::Error> for CryptoError {
    fn from(err: aes_gcm::Error) -> Self {
        CryptoError::AesGcm(err)
    }
}

impl From<std::string::FromUtf8Error> for CryptoError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        CryptoError::Utf8(err)
    }
}

impl From<bs58::decode::Error> for CryptoError {
    fn from(err: bs58::decode::Error) -> Self {
        CryptoError::Base58(err)
    }
}

impl From<libp2p::identity::ParseError> for CryptoError {
    fn from(err: libp2p::identity::ParseError) -> Self {
        CryptoError::PeerId(err)
    }
}

#[derive(NetworkBehaviour)]
#[behaviour(out_event = "ChatBehaviourEvent")]
struct ChatBehaviour {
    gossipsub: gossipsub::Behaviour,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
}

#[derive(Debug)]
enum ChatBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Kademlia(kad::Event),
}

impl From<gossipsub::Event> for ChatBehaviourEvent {
    fn from(event: gossipsub::Event) -> Self {
        ChatBehaviourEvent::Gossipsub(event)
    }
}

impl From<kad::Event> for ChatBehaviourEvent {
    fn from(event: kad::Event) -> Self {
        ChatBehaviourEvent::Kademlia(event)
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum ChatMessage {
    Chat {
        sender: String,
        content: Vec<u8>,
        nonce: Vec<u8>,
    },
}

fn derive_session_key(sender: &PeerId, receiver: &PeerId) -> [u8; 32] {
    let mut hasher = Sha256::new();
    let (first, second) = if sender < receiver {
        (sender, receiver)
    } else {
        (receiver, sender)
    };
    hasher.update(first.to_bytes());
    hasher.update(second.to_bytes());
    let result = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&result[..32]);
    key
}

fn encrypt_message(
    cipher: &Aes256Gcm,
    plaintext: &str,
    nonce: &[u8; 12],
) -> Result<Vec<u8>, CryptoError> {
    let ciphertext = cipher.encrypt(Nonce::from_slice(nonce), plaintext.as_bytes())?;
    Ok(ciphertext)
}

fn decrypt_message(
    cipher: &Aes256Gcm,
    ciphertext: &[u8],
    nonce: &[u8],
) -> Result<String, CryptoError> {
    let plaintext = cipher.decrypt(Nonce::from_slice(nonce), ciphertext)?;
    let plaintext_str = String::from_utf8(plaintext)?;
    Ok(plaintext_str)
}

fn get_or_create_key(path: &Path) -> Result<[u8; 32], Box<dyn Error>> {
    if path.exists() {
        println!("🔑 Found existing swarm key. Loading...");
        let key_bytes = fs::read(path)?;
        Ok(key_bytes
            .try_into()
            .map_err(|_| "Swarm key file must be exactly 32 bytes.")?)
    } else {
        println!("✨ No swarm key found. Generating a new one...");
        let key: [u8; 32] = rand::random();
        fs::write(path, key)?;
        println!("✅ Wrote new swarm key to: {:?}", path);
        Ok(key)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("✅ Local peer id: {local_peer_id}");

    let key_path = Path::new("swarm.key");
    let _key = get_or_create_key(key_path)?;

    let mut swarm = SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(5))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .mesh_n(4)
                .mesh_n_low(2)
                .mesh_outbound_min(2)
                .build()
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )?;

            let mut kademlia =
                kad::Behaviour::new(local_peer_id, kad::store::MemoryStore::new(local_peer_id));
            kademlia.set_mode(Some(kad::Mode::Server));

            Ok(ChatBehaviour {
                gossipsub,
                kademlia,
            })
        })?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    let chat_topic = gossipsub::IdentTopic::new("private-chat-room");
    swarm.behaviour_mut().gossipsub.subscribe(&chat_topic)?;

    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    swarm.listen_on("/ip4/0.0.0.0/tcp/4002".parse()?)?;

    let connected_peers = Arc::new(Mutex::new(HashMap::<PeerId, bool>::new()));
    let nonce_counters = Arc::new(Mutex::new(HashMap::<PeerId, u64>::new()));
    let known_peers = Arc::new(Mutex::new(HashSet::<PeerId>::new()));

    if let Some(addr_str) = std::env::args().nth(1) {
        let remote: Multiaddr = addr_str.parse()?;
        swarm.dial(remote.clone())?;
        println!("📞 Dialing bootstrap peer at {addr_str}");

        let remote_peer_id = remote
            .iter()
            .find_map(|p| match p {
                libp2p::multiaddr::Protocol::P2p(peer_id) => Some(peer_id),
                _ => None,
            })
            .ok_or("Bootstrap address must include a PeerId")?;
        swarm.behaviour_mut().kademlia.add_address(&remote_peer_id, remote);
        swarm.behaviour_mut().kademlia.bootstrap()?;
        known_peers.lock().await.insert(remote_peer_id);
    } else {
        println!("🔗 This is a bootstrap node. Waiting for others to connect...");
    }

    println!("🚀 Secure chat client started. Enter messages to send.");
    println!("ℹ️ For public IP, ensure port 4002 is forwarded on your router.");

    loop {
        tokio::select! {
            line = stdin.next_line() => {
                match line? {
                    Some(line) if !line.trim().is_empty() => {
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

                                let encrypted_content = match encrypt_message(&cipher, line.trim(), &nonce) {
                                    Ok(content) => content,
                                    Err(e) => {
                                        eprintln!("Encryption failed: {:?}", e);
                                        continue;
                                    }
                                };

                                let chat_msg = ChatMessage::Chat {
                                    sender: local_peer_id.to_string(),
                                    content: encrypted_content,
                                    nonce: nonce.to_vec(),
                                };
                                let json = serde_json::to_string(&chat_msg)?;

                                if let Err(e) = swarm
                                    .behaviour_mut()
                                    .gossipsub
                                    .publish(chat_topic.clone(), json.as_bytes())
                                {
                                    eprintln!("Error publishing message to {}: {:?}", peer_id, e);
                                } else {
                                    println!("📤 You (to {}): {}", peer_id, line.trim());
                                }
                            }
                        }
                    }
                    _ => {}
                }
                print!("> ");
                stdout().flush()?;
            }
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        let addr_str = address.with_p2p(local_peer_id).unwrap().to_string();
                        println!("👂 Listening on {}", addr_str);
                    }
                    SwarmEvent::Dialing { peer_id: Some(peer_id), .. } => {
                        println!("☎️ Dialing {peer_id}...");
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                        println!("🤝 Connection established with {peer_id}");
                        swarm
                            .behaviour_mut()
                            .kademlia
                            .add_address(&peer_id, endpoint.get_remote_address().clone());
                        connected_peers.lock().await.insert(peer_id, true);
                        known_peers.lock().await.insert(peer_id);
                    }
                    SwarmEvent::Behaviour(behaviour_event) => {
                        match behaviour_event {
                            ChatBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                                propagation_source: _peer_id,
                                message,
                                ..
                            }) => {
                                if let Ok(chat_msg) = serde_json::from_slice::<ChatMessage>(&message.data) {
                                    match chat_msg {
                                        ChatMessage::Chat { sender, content, nonce } => {
                                            // Decode the Base58-encoded PeerId string and parse it
                                            let sender_bytes = bs58::decode(&sender).into_vec()?;
                                            let sender_peer_id = match PeerId::from_bytes(&sender_bytes) {
                                                Ok(id) => id,
                                                Err(e) => {
                                                    eprintln!("Invalid sender PeerId: {:?}", e);
                                                    continue;
                                                }
                                            };

                                            let session_key = derive_session_key(&local_peer_id, &sender_peer_id);
                                            let cipher = match Aes256Gcm::new_from_slice(&session_key) {
                                                Ok(cipher) => cipher,
                                                Err(e) => {
                                                    eprintln!("Failed to create cipher: {:?}", e);
                                                    continue;
                                                }
                                            };

                                            let decrypted_content = match decrypt_message(&cipher, &content, &nonce) {
                                                Ok(content) => content,
                                                Err(e) => {
                                                    eprintln!("Decryption failed from {}: {:?}", sender, e);
                                                    continue;
                                                }
                                            };

                                            println!("\r📨 {}: {}", sender, decrypted_content);
                                        }
                                    }
                                }
                            }
                            ChatBehaviourEvent::Kademlia(kad_event) => {
                                log::debug!("[Kademlia Event]: {:?}", kad_event);
                            }
                            ChatBehaviourEvent::Gossipsub(gossip_event) => {
                                log::debug!("[Gossipsub Event]: {:?}", gossip_event);
                            }
                        }
                    }
                    SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                        println!("🔌 Connection lost with {peer_id}. Cause: {:?}", cause);
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