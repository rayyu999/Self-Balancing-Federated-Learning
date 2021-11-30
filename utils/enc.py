import random


def convert_ciphertext(sk1, pk1, pk2, ciphertext):
    r = random.randint(1, 1024)
    r_ciphertext = pk1.encrypt(r)
    mask = ciphertext + r_ciphertext
    plaintext = sk1.decrypt(mask)
    ciphertext = pk2.encrypt(plaintext)
    return ciphertext - r_ciphertext
