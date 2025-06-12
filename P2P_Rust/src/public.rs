use futures::StreamExt;
use libp2p::{
    gossipsub,
    identity,
    mdns,
    noise,
    relay,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp,
    yamux,
    PeerId,
    SwarmBuilder,
};
use std::error::Error;
use tokio::{
    io::{self, AsyncBufReadExt, AsyncWriteExt},
    select,
};

// Custom network behaviour combining Gossipsub, Mdns, and Relay
#[derive(NetworkBehaviour)]
#[behaviour(out_event = "ChatBehaviourEvent")]
struct ChatBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    relay: relay::Behaviour,
}

#[derive(Debug)]
enum ChatBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Mdns(mdns::Event),
    Relay(relay::Event),
}

// Implement From traits for ChatBehaviourEvent to satisfy NetworkBehaviour
impl From<gossipsub::Event> for ChatBehaviourEvent {
    fn from(event: gossipsub::Event) -> Self {
        ChatBehaviourEvent::Gossipsub(event)
    }
}

impl From<mdns::Event> for ChatBehaviourEvent {
    fn from(event: mdns::Event) -> Self {
        ChatBehaviourEvent::Mdns(event)
    }
}

impl From<relay::Event> for ChatBehaviourEvent {
    fn from(event: relay::Event) -> Self {
        ChatBehaviourEvent::Relay(event)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Create a random PeerId
    let id_keys = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(id_keys.public());
    println!("✅ Local peer id: {local_peer_id}");

    // Create a Gossipsub topic
    let topic = gossipsub::IdentTopic::new("chat-room");

    // Create a Swarm to manage peers and events
    let behaviour = ChatBehaviour {
        gossipsub: gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(id_keys.clone()),
            gossipsub::Config::default(),
        )?,
        mdns: mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?,
        relay: relay::Behaviour::new(local_peer_id, relay::Config::default()),
    };

    let mut swarm = SwarmBuilder::with_existing_identity(id_keys)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|_| behaviour)?
        .build();

    // Listen on all interfaces with a fixed port for public IP access
    swarm.listen_on("/ip4/0.0.0.0/tcp/4001".parse()?)?;

    // Connect to a bootstrap node if provided (for public IP or relay)
    if let Some(addr) = std::env::args().nth(1) {
        let remote: libp2p::Multiaddr = addr.parse()?;
        swarm.dial(remote.clone())?;
        println!("📞 Dialed bootstrap peer at {addr}");
    }

    let mut stdin = io::BufReader::new(io::stdin()).lines();

    // Subscribe to the chat topic
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    println!("🚀 Chat client started. Enter messages to send.");
    println!("ℹ️ For public IP, ensure port 4001 is forwarded on your router.");

    // The main event loop
    loop {
        select! {
            // Handle user input from the terminal
            line = stdin.next_line() => {
                let line = line?.expect("stdin closed");
                if let Err(e) = swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(topic.clone(), line.as_bytes())
                {
                    println!("❌ Publish error: {e:?}");
                }
            }
            // Handle events from the swarm
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("👂 Listening on {address}");
                    }
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source: peer_id,
                        message,
                        ..
                    })) => {
                        print!("\r                       \r"); // Clear line
                        println!(
                            "📨 {}: {}",
                            peer_id,
                            String::from_utf8_lossy(&message.data),
                        );
                        print!("> ");
                        tokio::io::stdout().flush().await?;
                    }
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                        for (peer_id, _multiaddr) in list {
                            println!("[mdns] Discovered peer: {peer_id}");
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                        }
                    }
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                        for (peer_id, _multiaddr) in list {
                            println!("[mdns] Expired peer: {peer_id}");
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                        }
                    }
                    SwarmEvent::Behaviour(ChatBehaviourEvent::Relay(relay::Event::ReservationReqAccepted { .. })) => {
                        println!("[relay] Reservation request accepted");
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        println!("🤝 Connection established with {peer_id}");
                    }
                    _ => {}
                }
            }
        }
    }
}