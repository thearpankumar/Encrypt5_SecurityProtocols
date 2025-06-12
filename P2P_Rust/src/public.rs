use futures::StreamExt;
use libp2p::{
    gossipsub, identity, kad, noise,
    // The NetworkBehaviour derive macro is now brought in via the "macros" feature
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::time::Duration;
// FIX: We need `tokio::io` for `stdin` and `AsyncBufReadExt` for `.lines()`
use tokio::io::{self, AsyncBufReadExt};
use tokio::select;

// A simple message struct that we'll serialize and send.
// libp2p's security layer will encrypt this for us automatically.
#[derive(Serialize, Deserialize, Debug)]
struct ChatMessage {
    sender: String,
    content: String,
    sequence_number: u64,
}

// FIX: This derive will now work because of the "macros" feature in Cargo.toml
#[derive(NetworkBehaviour)]
// The derive macro will generate a `ChatBehaviourEvent` enum for us.
// We don't need to define it manually.
struct ChatBehaviour {
    gossipsub: gossipsub::Behaviour,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("✅ Local peer id: {local_peer_id}");

    let mut swarm = SwarmBuilder::with_existing_identity(local_key.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        // FIX: The closure now directly returns the behaviour struct, not a Result.
        // This satisfies the trait bounds because the operations inside use `.expect()`.
        .with_behaviour(|key| {
            let message_authenticity = gossipsub::MessageAuthenticity::Signed(key.clone());

            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(10))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .build()
                .expect("Valid gossipsub config");

            let gossipsub = gossipsub::Behaviour::new(message_authenticity, gossipsub_config)
                .expect("Couldn't build gossipsub");

            let kademlia =
                kad::Behaviour::new(local_peer_id, kad::store::MemoryStore::new(local_peer_id));

            // Directly return the struct.
            ChatBehaviour { gossipsub, kademlia }
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    let topic = gossipsub::IdentTopic::new("public-chat");
    // FIX: This now works because the derive macro created the `.gossipsub` field.
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    if let Some(addr_str) = std::env::args().nth(1) {
        let remote: Multiaddr = addr_str.parse()?;
        swarm.dial(remote)?;
        println!("📞 Dialed bootstrap peer at {addr_str}");
    }

    println!("🚀 Secure chat client started. Enter messages to send.");

    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    let mut sequence_number: u64 = 0;

    loop {
        select! {
            line = stdin.next_line() => {
                let line = line?.unwrap_or_default();
                if line.is_empty() { continue; }

                sequence_number += 1;
                let chat_message = ChatMessage {
                    sender: local_peer_id.to_string(),
                    content: line,
                    sequence_number,
                };

                let json = serde_json::to_string(&chat_message)?;

                if let Err(e) = swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(topic.clone(), json.as_bytes())
                {
                    eprintln!("❌ Publish error: {e:?}");
                }
            },
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("👂 Listening on {}", address.with_p2p(local_peer_id).unwrap());
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        println!("🤝 Connection established with {peer_id}");
                    }
                    // FIX: This now works because the derive macro generated `ChatBehaviourEvent`.
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source: peer_id,
                        message,
                        ..
                    })) => {
                        match serde_json::from_slice::<ChatMessage>(&message.data) {
                            Ok(msg) => {
                                println!(
                                    "📨 [{}] (from {}): {}",
                                    msg.sender,
                                    peer_id,
                                    msg.content
                                );
                            }
                            Err(_) => {
                                println!(
                                    "📨 Received raw message from {}: {}",
                                    peer_id,
                                    String::from_utf8_lossy(&message.data)
                                );
                            }
                        }
                    }
                    SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                        println!("🔌 Connection lost with {peer_id}: {:?}", cause);
                    }
                    _ => {}
                }
            }
        }
    }
}