import os
import base64
from pathlib import Path
from etsi_qkd_014_client import QKD014Client
from etsi_qkd_014_client.data import DataKeyContainer, DataStatus, DataError

"""
QuKayDee Configuration & Sample Output
======================================

KME Configuration:
- KME-1 (Alice): kme-1.acct-2764.etsi-qkd-api.qukaydee.com
- KME-2 (Bob):   kme-2.acct-2764.etsi-qkd-api.qukaydee.com

SAE Configuration:
- SAE-1 (Alice): Uses sae-1.pem certificate, connects to KME-1
- SAE-2 (Bob):   Uses sae-2.pem certificate, connects to KME-2

Key Stream Status:
- Key Rate: 1000 bits/sec
- Key Expiry Time: 60 seconds
- Stored Key Bits: 1,000,000
- Stored Key Count: 1953
- Max Key Per Request: 100
- Max Key Size: 100000 bits
- Min Key Size: 1 bit

Sample Output:
--------------
DEMO 1 - Status Response:
  source_KME_ID: kme-1
  target_KME_ID: kme-2
  master_SAE_ID: sae-1
  slave_SAE_ID: sae-2
  key_size: 512 bits
  stored_key_count: 1953

DEMO 1 - Key Retrieval:
  Key ID: 0689504c-b267-44b6-b1f9-9f5fc052621f
  Key Size: 32 bytes (256 bits)
  Key (base64): Z+c1hqbzA7GW0Z/1jwBlTdKohjJeRPQfKu4s2gWpSuQ=

DEMO 2 - Peer-to-Peer QKD:
  Shared Key ID: fb5937fc-7718-455d-ab72-da0863ce93fb
  Key Matches: True (verified between Alice and Bob)

DEMO 3 - Encryption Example:
  Key ID: ba72268c-5605-4e90-bac9-9ba7de7e80a5
  Plaintext: This is a secret message protected by quantum cryptography!
  Ciphertext (hex): ef9863914d591d15511269a4c9c9a9837f0df2f38e688172234b366682fa7097...
  IV (hex): 52357cf196d52500b648510533184535
  Decryption: SUCCESS 
"""


class QKDClientException(Exception):
    """Custom exception for QKD client errors."""

    pass


class QuKayDeeClient:
    """Wrapper class for interacting with QuKayDee QKD servers."""

    def __init__(
        self, sae_id: str, kme_hostname: str = None, force_insecure: bool = False
    ):
        """
        Initialize QuKayDee client.

        Args:
            sae_id: Security Application Entity ID (e.g., 'sae-1' or 'sae-2')
            kme_hostname: Hostname or IP address of the KME (default: auto-detect from sae_id)
            force_insecure: Skip SSL verification (default: False - uses proper certificate verification)
        """
        self.sae_id = sae_id

        # Auto-detect KME hostname based on SAE ID
        if kme_hostname is None:
            # Map sae-1 -> kme-1, sae-2 -> kme-2
            kme_id = sae_id.replace("sae-", "kme-")
            self.kme_hostname = f"{kme_id}.acct-2764.etsi-qkd-api.qukaydee.com"
        else:
            self.kme_hostname = kme_hostname

        self.force_insecure = force_insecure

        # Certificate paths (try PEM first, then separate cert/key)
        pem_file = f"{sae_id}.pem"
        if Path(pem_file).exists():
            # Use combined PEM file
            self.cert_file = pem_file
            self.key_file = pem_file
        else:
            # Use separate cert and key files
            self.cert_file = f"{sae_id}.crt"
            self.key_file = f"{sae_id}.key"

        # Use the QuKayDee server CA certificate for proper SSL verification
        ca_files = list(Path(".").glob("*-server-ca-qukaydee-com.crt"))
        if not ca_files:
            raise FileNotFoundError(
                "Server CA certificate ('*-server-ca-qukaydee-com.crt') not found."
            )
        if len(ca_files) > 1:
            print(f"Warning: Multiple server CA files found, using {ca_files[0]}")
        self.ca_file = str(ca_files[0])

        # Verify certificate files exist
        self._verify_certificates()

        # Initialize QKD client
        self.client = None

    def _verify_certificates(self):
        """Verify that all required certificate files exist."""
        required_files = [self.cert_file, self.key_file, self.ca_file]

        for file in required_files:
            if not Path(file).exists():
                raise FileNotFoundError(f"Certificate file not found: {file}")
        print(f" All certificates found for {self.sae_id}")

    def connect(self):
        """Establish connection to QuKayDee QKD server."""
        try:
            # Create QKD client with certificates and proper SSL verification
            self.client = QKD014Client(
                kme_hostname=self.kme_hostname,
                cert_path=self.cert_file,
                key_path=self.key_file,
                ca_path=self.ca_file,
                force_insecure=self.force_insecure,
            )
            if self.force_insecure:
                print(f"⚠ SSL verification disabled")
            else:
                print(f" Secure connection established (SSL verified)")
            print(f" Connected to QuKayDee as {self.sae_id}")
            return True
        except Exception as e:
            print(f" Connection failed: {e}")
            raise QKDClientException(f"Failed to connect: {e}")

    def get_status(self, slave_sae_id: str):
        """
        Get the status of the QKD server.

        Args:
            slave_sae_id: URL-encoded SAE ID of slave SAE

        Returns:
            DataStatus: Status information from the QKD server
        """
        if not self.client:
            raise QKDClientException("Client not connected. Call connect() first.")

        try:
            status_code, status_data = self.client.get_status(slave_sae_id)

            if status_code != 200:
                print(f" Status request failed with code {status_code}")
                if isinstance(status_data, DataError):
                    raise QKDClientException(f"Error: {status_data}")

            print(f" Status retrieved for {self.sae_id}")
            return status_data
        except Exception as e:
            print(f" Failed to get status: {e}")
            raise

    def get_key(self, slave_sae_id: str, size: int = None, number: int = None):
        """
        Request a quantum key from the QKD server.

        Args:
            slave_sae_id: URL-encoded SAE ID of slave SAE (required)
            size: Key size in bits (optional, uses server default if None)
            number: Number of keys requested (optional, default is 1)

        Returns:
            dict: Key information with 'key_ID' and 'key' fields
        """
        if not self.client:
            raise QKDClientException("Client not connected. Call connect() first.")

        try:
            # Request key with specified parameters
            status_code, key_container = self.client.get_key(
                slave_sae_id=slave_sae_id, size=size, number=number
            )

            if status_code != 200:
                print(f" Key request failed with code {status_code}")
                if isinstance(key_container, DataError):
                    raise QKDClientException(f"Error: {key_container}")

            # Extract the first key from the container
            if not key_container.keys:
                raise QKDClientException("No keys returned from server")

            first_key = key_container.keys[0]
            print(f" Key retrieved: Key ID: {first_key.key_id}")

            # Return a dict for compatibility with demo code
            return {"key_ID": first_key.key_id, "key": first_key.key}
        except Exception as e:
            print(f" Failed to get key: {e}")
            raise

    def get_key_with_id(self, master_sae_id: str, key_id: str):
        """
        Retrieve a specific key using its ID.

        Args:
            master_sae_id: URL-encoded SAE ID of master SAE
            key_id: The UUID of the key to retrieve

        Returns:
            dict: Key information with 'key_ID' and 'key' fields
        """
        if not self.client:
            raise QKDClientException("Client not connected. Call connect() first.")

        try:
            # get_key_with_key_IDs expects a list of key IDs
            status_code, key_container = self.client.get_key_with_key_IDs(
                master_sae_id=master_sae_id, key_ids=[key_id]
            )

            if status_code != 200:
                print(f" Key retrieval failed with code {status_code}")
                if isinstance(key_container, DataError):
                    raise QKDClientException(f"Error: {key_container}")

            # Extract the first key from the container
            if not key_container.keys:
                raise QKDClientException("No keys returned from server")

            first_key = key_container.keys[0]
            print(f" Key retrieved with ID: {key_id}")

            # Return a dict for compatibility with demo code
            return {"key_ID": first_key.key_id, "key": first_key.key}
        except Exception as e:
            print(f" Failed to get key with ID: {e}")
            raise


def demonstrate_qkd_basic():
    """Basic QKD demonstration with a single SAE."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic QKD Key Retrieval")
    print("=" * 60)

    try:
        # Initialize client for SAE-1 (Alice)
        qkd_client = QuKayDeeClient(sae_id="sae-1")

        # Connect to QuKayDee
        qkd_client.connect()

        # Get server status (requires specifying the slave SAE)
        print("\n--- Server Status ---")
        status = qkd_client.get_status(slave_sae_id="sae-2")
        print(f"Status: {status}")

        # Request a 256-bit quantum key with sae-2 (Bob)
        print("\n--- Key Retrieval ---")
        key_data = qkd_client.get_key(slave_sae_id="sae-2", size=256)
        print(f"Key ID: {key_data.get('key_ID')}")
        # Key is base64 encoded, decode to get actual bytes
        key_bytes = base64.b64decode(key_data.get("key", ""))
        print(f"Key Size: {len(key_bytes)} bytes")
        print(f"Key (base64): {key_data.get('key', '')[:64]}...")

        return key_data

    except Exception as e:
        print(f"Demo failed: {e}")
        return None


def demonstrate_qkd_peer_to_peer():
    """Demonstrate QKD between two SAEs (Alice ↔ Bob)."""
    print("\n" + "=" * 60)
    print("DEMO 2: Peer-to-Peer QKD (Alice ↔ Bob)")
    print("=" * 60)

    try:
        # Initialize both SAE clients
        alice = QuKayDeeClient(sae_id="sae-1")  # Alice uses sae-1 certificates
        bob = QuKayDeeClient(sae_id="sae-2")  # Bob uses sae-2 certificates

        # Connect both clients
        alice.connect()
        bob.connect()

        # Alice requests a shared key with Bob
        print("\n--- Alice (sae-1): Requesting shared key with Bob (sae-2) ---")
        key1 = alice.get_key(slave_sae_id="sae-2", size=256)

        # Bob retrieves the same key using the key ID
        print("\n--- Bob (sae-2): Retrieving shared key ---")
        key_id = key1.get("key_ID")
        # When retrieving with key ID, Bob is the slave, so master is Alice
        key2 = bob.get_key_with_id(master_sae_id="sae-1", key_id=key_id)

        # Verify both keys match
        if key1.get("key") == key2.get("key"):
            print(" SUCCESS: Both Alice and Bob have the same quantum key!")
            print(f"\nShared Key ID: {key_id}")
            print(f"Key matches: {key1.get('key') == key2.get('key')}")
        else:
            print(" ERROR: Keys do not match!")

        return key1, key2

    except Exception as e:
        print(f"Peer-to-peer demo failed: {e}")
        return None, None


def demonstrate_qkd_encryption():
    """Demonstrate using QKD keys for encryption between two parties."""
    print("\n" + "=" * 60)
    print("DEMO 3: Using QKD Keys for Encryption (Alice ↔ Bob)")
    print("=" * 60)

    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        import secrets

        # 1. Initialize clients for both Alice and Bob
        alice = QuKayDeeClient(sae_id="sae-1")
        bob = QuKayDeeClient(sae_id="sae-2")
        alice.connect()
        bob.connect()

        # 2. Alice requests a shared key and uses it to encrypt a message
        print("\n--- Alice: Requesting shared key and encrypting message ---")
        alice_key_data = alice.get_key(slave_sae_id="sae-2", size=256)
        alice_key = base64.b64decode(alice_key_data.get("key"))
        key_id = alice_key_data.get("key_ID")

        plaintext = b"This is a secret message protected by quantum cryptography!"
        iv = secrets.token_bytes(16)

        if len(alice_key) < 32:
            raise QKDClientException(f"Key too short: {len(alice_key)} bytes, need 32")

        cipher_alice = Cipher(
            algorithms.AES(alice_key[:32]),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher_alice.encryptor()
        padded = plaintext + b" " * (16 - len(plaintext) % 16)
        ciphertext = encryptor.update(padded) + encryptor.finalize()

        print(" Alice encrypted the message successfully.")
        print(f"Key ID used: {key_id}")
        print(f"Plaintext: {plaintext.decode()}")
        print(f"Ciphertext (hex): {ciphertext.hex()}")

        # 3. Bob retrieves the same key using the Key ID
        print("\n--- Bob: Retrieving shared key for decryption ---")
        bob_key_data = bob.get_key_with_id(master_sae_id="sae-1", key_id=key_id)
        bob_key = base64.b64decode(bob_key_data.get("key"))

        # Verify Alice and Bob have the same key
        if alice_key != bob_key:
            print(" CRITICAL ERROR: Alice's and Bob's keys do not match!")
            return
        else:
            print(" Bob has successfully retrieved the same quantum key.")

        # 4. Bob uses his key to decrypt the message
        print("\n--- Bob: Decrypting message from Alice ---")
        cipher_bob = Cipher(
            algorithms.AES(bob_key[:32]),
            modes.CBC(iv),  # Bob needs the same IV
            backend=default_backend(),
        )
        decryptor = cipher_bob.decryptor()
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
        decrypted = decrypted_padded.strip()

        # 5. Verification
        print(f" Bob decrypted the message successfully.")
        print(f"Decrypted message: {decrypted.decode()}")

        if decrypted == plaintext:
            print("\n SUCCESS: Decrypted message matches original plaintext!")
        else:
            print("\n FAILURE: Decrypted message does not match original plaintext!")

    except ImportError:
        print("cryptography library not installed. Skipping encryption demo.")
        print("Install with: pip install cryptography")
    except Exception as e:
        print(f"Encryption demo failed: {e}")


def demonstrate_key_destruction():
    """Demonstrate that a key is 'destroyed' after a single retrieval."""
    print("\n" + "=" * 60)
    print("DEMO 4: Key Destruction after Eavesdropping")
    print("=" * 60)
    print("Scenario: Eve steals Bob's credentials to intercept the key.")
    print("Hypothesis: A key can only be retrieved once. Bob's attempt should fail.")

    try:
        # 1. Initialize clients for all parties
        alice = QuKayDeeClient(sae_id="sae-1")
        # "Eve" will use Bob's credentials to impersonate him
        eve_as_bob = QuKayDeeClient(sae_id="sae-2")
        # The real Bob
        bob = QuKayDeeClient(sae_id="sae-2")

        # 2. Alice requests a shared key for Bob
        print("\n--- Alice: Requesting a shared key for Bob ---")
        alice.connect()
        alice_key_data = alice.get_key(slave_sae_id="sae-2", size=256)
        key_id = alice_key_data.get("key_ID")
        print(f"Alice requested key with ID: {key_id} for Bob.")

        # 3. Eve (impersonating Bob) intercepts the key first
        print("\n--- Eve's Interception (as Bob) ---")
        eve_as_bob.connect()
        eve_key_data = eve_as_bob.get_key_with_id(master_sae_id="sae-1", key_id=key_id)
        print("Eve successfully intercepted the key by impersonating Bob.")
        print(f"Intercepted Key (base64): {eve_key_data.get('key')[:32]}...")


        # 4. The real Bob now tries to retrieve the same key
        print("\n--- Bob's Attempt (Legitimate User) ---")
        try:
            bob.connect()
            # This call should fail if the key is single-use
            bob.get_key_with_id(master_sae_id="sae-1", key_id=key_id)
            # If this line is reached, the hypothesis is wrong
            print("\nDEMO RESULT: FAILURE")
            print("Bob was able to retrieve the key, meaning it is NOT single-use.")
            print("The eavesdropping was not detected by this mechanism.")

        except QKDClientException as e:
            # We expect an exception here, which proves the key was destroyed.
            print("\nDEMO RESULT: SUCCESS")
            print("Bob's attempt to get the key failed as expected.")
            print("This demonstrates the key was 'destroyed' after Eve's interception.")
            print(f"Server Error: {e}")

    except Exception as e:
        print(f"Key destruction demo failed: {e}")


def main():
    """Main entry point for QKD demonstrations."""
    print("QuKayDee QKD Client - Sample Implementation")
    print("Using ETSI QKD 014 Standard\n")

    try:
        # Demo 1: Basic key retrieval
        demonstrate_qkd_basic()

        # Demo 2: Peer-to-peer QKD
        demonstrate_qkd_peer_to_peer()

        # Demo 3: Using QKD keys for encryption
        demonstrate_qkd_encryption()

        # Demo 4: Key Destruction after Eavesdropping
        demonstrate_key_destruction()

        print("\n" + "=" * 60)
        print("All demonstrations completed!")
        print("=" * 60)

    except Exception as e:
        print(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
