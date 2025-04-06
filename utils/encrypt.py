import base64
import json
import os
from hashlib import pbkdf2_hmac
from Crypto.Cipher import AES

def aes_encrypt(plain_text, passphrase="BGILyPIS"):
    # Generate a random salt and IV
    salt = os.urandom(256)
    iv = os.urandom(16)
    
    # Derive the key using PBKDF2 (sha512)
    key = pbkdf2_hmac("sha512", passphrase.encode(), salt, 999, 32)
    
    # Padding to make text a multiple of 16 bytes (PKCS7)
    block_size = 16
    pad_len = block_size - (len(plain_text) % block_size)
    padded_text = plain_text + chr(pad_len) * pad_len
    
    # Encrypt using AES in CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted_data = cipher.encrypt(padded_text.encode())
    
    # Prepare the data to return
    data = {
        "ciphertext": base64.b64encode(encrypted_data).decode(),
        "iv": iv.hex(),
        "salt": salt.hex()
    }

    # Return the base64-encoded data
    return base64.b64encode(json.dumps(data).encode()).decode()


def aes_decrypt(encrypted_text, passphrase="BGILyPIS"):
    # Decode base64-encoded JSON data
    json_data = json.loads(base64.b64decode(encrypted_text).decode())
    
    # Extract the IV, salt, and ciphertext
    iv = bytes.fromhex(json_data['iv'])
    salt = bytes.fromhex(json_data['salt'])
    ciphertext = base64.b64decode(json_data['ciphertext'])
    
    # Derive the key using PBKDF2 (same as encryption)
    key = pbkdf2_hmac("sha512", passphrase.encode(), salt, 999, 32)
    
    # Decrypt the ciphertext using AES CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext).decode()

    # Remove padding
    pad_len = ord(decrypted_padded[-1])
    decrypted_text = decrypted_padded[:-pad_len]
    return decrypted_text