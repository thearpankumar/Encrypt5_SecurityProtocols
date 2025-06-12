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
    collections::HashMap,
    error::Error,
    fs,
    io::{stdout, Write},
    path::Path,
    sync::Arc,
};
use tokio::{io::AsyncBufReadExt, sync::Mutex, time::Duration};

// Custom network behaviour
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
    Chat { sender: String, content: String },
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
                .heartbeat_interval(Duration::from_secs(10))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .mesh_n_low(0)
                .mesh_outbound_min(0)
                .build()
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )?;

            let kademlia =
                kad::Behaviour::new(local_peer_id, kad::store::MemoryStore::new(local_peer_id));

            Ok(ChatBehaviour {
                gossipsub,
                kademlia,
            })
        })?
        .build();

    let chat_topic = gossipsub::IdentTopic::new("private-chat-room");
    swarm.behaviour_mut().gossipsub.subscribe(&chat_topic)?;

    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    swarm.listen_on("/ip4/0.0.0.0/tcp/4001".parse()?)?;

    if let Some(addr_str) = std::env::args().nth(1) {
        let remote: Multiaddr = addr_str.parse()?;
        swarm.dial(remote)?;
        println!("📞 Dialing bootstrap peer at {addr_str}");
    } else {
        println!("🔗 This is a bootstrap node. Waiting for others to connect...");
    }

    println!("🚀 Secure chat client started. Enter messages to send.");
    println!("ℹ️ For public IP, ensure port 4001 is forwarded on your router.");

    let connected_peers = Arc::new(Mutex::new(HashMap::<PeerId, bool>::new()));

    loop {
        tokio::select! {
            line = stdin.next_line() => {
                match line? {
                    Some(line) if !line.trim().is_empty() => {
                        let chat_msg = ChatMessage::Chat {
                            sender: local_peer_id.to_string(),
                            content: line.trim().to_string(),
                        };
                        let json = serde_json::to_string(&chat_msg)?;
                        let mut peers = connected_peers.lock().await;

                        if peers.is_empty() {
                            println!("⚠️ No peers connected. Cannot send message.");
                        } else {
                            if let Err(e) = swarm
                                .behaviour_mut()
                                .gossipsub
                                .publish(chat_topic.clone(), json.as_bytes())
                            {
                                eprintln!("Error publishing message: {:?}", e);
                            } else {
                                println!("📤 You: {}", line.trim());
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
                    }
                    SwarmEvent::Behaviour(behaviour_event) => {
                        match behaviour_event {
                            ChatBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                                propagation_source: peer_id,
                                message,
                                ..
                            }) => {
                                if let Ok(chat_msg) = serde_json::from_slice::<ChatMessage>(&message.data) {
                                    match chat_msg {
                                        ChatMessage::Chat { sender, content } => {
                                            println!("\r📨 {}: {}", sender, content);
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
                    SwarmEvent::ConnectionClosed { peer_id, .. } => {
                        println!("🔌 Connection lost with {peer_id}");
                        connected_peers.lock().await.remove(&peer_id);
                    }
                    _ => {}
                }
                print!("\r> ");
                stdout().flush()?;
            }
        }
    }
}