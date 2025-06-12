// src/main.rs

use futures::StreamExt;
use libp2p::{
    gossipsub, identity, mdns, noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
};
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::hash::{Hash, Hasher};
use tokio::io::{self, AsyncBufReadExt};

// The NetworkBehaviour struct remains the same.
// The derive macro will generate a `ChatBehaviourEvent` enum for us.
#[derive(NetworkBehaviour)]
struct ChatBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("✅ Local peer id: {local_peer_id}");

    // --- FIX: This is the new, fluent builder API ---
    let mut swarm = SwarmBuilder::with_existing_identity(local_key.clone()) // Use the keypair
        .with_tokio() // Tell libp2p to use the Tokio runtime
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new, // Use the noise protocol for encryption
            yamux::Config::default, // Use the yamux protocol for multiplexing
        )?
        .with_behaviour(|key| {
            // This closure is called to create the behaviour.
            // `key` is the same keypair we passed to `with_existing_identity`.

            // Set up Gossipsub
            let message_id_fn = |message: &gossipsub::Message| {
                let mut s = DefaultHasher::new();
                message.data.hash(&mut s);
                gossipsub::MessageId::from(s.finish().to_string())
            };
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .message_id_fn(message_id_fn)
                .build()
                .expect("Valid gossipsub config");

            // FIX: `MessageAuthenticity::Signed` takes ownership of the key.
            // We need to clone it because the SwarmBuilder also has a copy.
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )
                .expect("Correct gossipsub");

            // Set up mDNS for peer discovery on the local network
            let mdns =
                mdns::tokio::Behaviour::new(mdns::Config::default(), key.public().to_peer_id())?;

            Ok(ChatBehaviour { gossipsub, mdns })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(std::time::Duration::from_secs(60)))
        .build();
    // --- End of the new Swarm build process ---

    // The rest of the code is largely the same.
    let topic = gossipsub::IdentTopic::new("chat-room");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    let mut stdin = io::BufReader::new(io::stdin()).lines();
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    if let Some(addr_str) = std::env::args().nth(1) {
        let remote: Multiaddr = addr_str.parse()?;
        swarm.dial(remote)?;
        println!("📞 Dialed bootstrap peer at {addr_str}");
    }

    println!("🚀 Chat client started. Enter messages to send.");

    loop {
        tokio::select! {
            line = stdin.next_line() => {
                 match line? {
                    Some(line) => {
                        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic.clone(), line.as_bytes()) {
                            println!("❌ Publish error: {e:?}");
                        }
                    },
                    None => { println!("👋 stdin closed, exiting."); break; }
                }
            },
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("👂 Listening on {address}");
                    },
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source: peer_id, message, ..
                    })) => {
                        println!("📨 {}: {}", peer_id, String::from_utf8_lossy(&message.data));
                    },
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                        for (peer_id, _multiaddr) in list {
                            println!("[mdns] Discovered: {peer_id}");
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                        }
                    },
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                        for (peer_id, _multiaddr) in list {
                            println!("[mdns] Expired: {peer_id}");
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                        }
                    },
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        println!("🤝 Connection established with {peer_id}");
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}