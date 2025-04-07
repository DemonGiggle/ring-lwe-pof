#
# Paper: On Ideal Lattices and Learning with Errors Over Rings
#
# Author: Vadim Lyubashevsky, Chris Peikert, Oded Regev [2013]
#
# In the implement below, the error distribution is random choosen from {0, 1}
# for the sake of easy implementation.
#
# This choice will result in severe security fault. The error distribution is
# a very critical part for the security.
import numpy as np
from scipy.stats import norm
import math

VERBOSE = 0

#
# Define a quotient ring as R_q = Z_q[x]/I
#

# The order of the quotient ring is defined as:
# n should be power of 2 (cyclotomic polynomial)
n = pow(2, 10)  # Increased from 2^9 to 2^10 for better security

# The ideal in the quotient ring is defined as below
# I = <x^n+1>
I = np.poly1d([1] + [0]*(n-1) + [1])

# q should be a prime and q = 1 mod 2n for NTT efficiency
# Finding such q that is congruent to 1 modulo 2n
def find_appropriate_q(n):
    # Start with 2n
    k = 2 * n
    while True:
        q = k + 1
        # Check if q is prime using probabilistic primality test
        if is_probable_prime(q) and q > n*n*10:  # q should be sufficiently large
            return q
        k += 2 * n

# Simple Miller-Rabin primality test
def is_probable_prime(n, k=40):
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = np.random.randint(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# Find appropriate q
q = find_appropriate_q(n)
print(f"Using n = {n}, q = {q}")

# Discrete Gaussian Sampler for error distribution
def discrete_gaussian_sampler(size, sigma):
    """
    Sample from a discrete Gaussian distribution centered at 0
    with standard deviation sigma, returning integer values.
    """
    # Sample from continuous Gaussian
    samples = np.random.normal(0, sigma, size)
    # Round to nearest integer and return
    return np.round(samples).astype(np.int64)

# Security parameter - standard deviation for error distribution
# Higher sigma increases security but also increases error probability
sigma = 3.2  # Standard choice for medium security level

# Now we generate a keypair
#
# Secret key s <- R_q with coefficients sampled from discrete Gaussian
s_coeffs = discrete_gaussian_sampler(n, sigma)
s = np.poly1d(np.mod(s_coeffs, q))

# Generate an error e <- R_q with coefficients sampled from discrete Gaussian
e_coeffs = discrete_gaussian_sampler(n, sigma)
e = np.poly1d(np.mod(e_coeffs, q))

# The public key pair is a tuple (a, b), where
# a <- R_q choosing uniformly random, and b = a * s + e (mod q, x^n+1)
a = np.poly1d(np.random.randint(0, q, n))

# Compute b = a*s + e (mod q, x^n+1)
_, b = np.polydiv(a * s, I)
b = np.poly1d(np.mod(b.coeffs + e.coeffs, q))

# Helper function for polynomial reduction modulo I and q
def poly_mod(poly, modulus_poly, q):
    _, remainder = np.polydiv(poly, modulus_poly)
    # Ensure coefficients are in [0, q)
    return np.poly1d(np.mod(remainder.coeffs, q))

#
# Start to encrypt and decrypt a message
#

# To avoid padding, we create a message just fit into n-bit
m = ''
for i in range(int(n / 8)):
    m = m + chr(ord('a') + i % 26)

# Convert message into binary array and then to polynomial
m_bin = [int(bit) for char in m for bit in format(ord(char), '08b')]
# Pad or truncate to match n
if len(m_bin) < n:
    m_bin = m_bin + [0] * (n - len(m_bin))
elif len(m_bin) > n:
    m_bin = m_bin[:n]

m_poly = np.poly1d(m_bin)

# Now, we start to encrypt the message m to (u, v) <- R_q
# where,
#   u = a * r + e1 (mod q, x^n+1)
#   v = b * r + e2 + [q/2] * m (mod q, x^n+1)
#
# Sample r, e1, e2 from the discrete Gaussian distribution
r_coeffs = discrete_gaussian_sampler(n, sigma)
r = np.poly1d(np.mod(r_coeffs, q))

e1_coeffs = discrete_gaussian_sampler(n, sigma)
e1 = np.poly1d(np.mod(e1_coeffs, q))

e2_coeffs = discrete_gaussian_sampler(n, sigma)
e2 = np.poly1d(np.mod(e2_coeffs, q))

# Compute u = a*r + e1 (mod q, x^n+1)
u = poly_mod(a * r + e1, I, q)

# Compute v = b*r + e2 + [q/2]*m (mod q, x^n+1)
# For encoding the message, we use q/2 for bit 1
m_encoded = np.poly1d(np.mod(np.array(m_bin) * (q // 2), q))
v = poly_mod(b * r + e2 + m_encoded, I, q)

# Now, let's decode it.
#
# The decode formula is:
#   v - u * s = (b*r + e2 + [q/2]*m) - (a*r + e1)*s
#              = (a*s + e)*r + e2 + [q/2]*m - a*r*s - e1*s
#              = e*r + e2 - e1*s + [q/2]*m
#
# The error term (e*r + e2 - e1*s) should be small relative to q/4
# for successful decoding.

# Compute v - u*s (mod q, x^n+1)
us = poly_mod(u * s, I, q)
d_poly = np.poly1d(np.mod(v.coeffs - us.coeffs, q))

# Now check each coefficient to see whether they are near 0 or q/2
# Output 0 if it is near 0, otherwise output 1
coeffs = d_poly.coeffs

# poly1d.coeffs will trim the leading zeros
# Ensure we have exactly n coefficients
if len(coeffs) < n:
    coeffs = np.concatenate((np.zeros(n - len(coeffs)), coeffs))
elif len(coeffs) > n:
    coeffs = coeffs[-n:]

# Decoding logic: if coefficient is closer to q/2 than to 0 or q, then it's a 1
# We consider 3 regions: [0, q/4), [q/4, 3q/4), [3q/4, q)
# Values in [q/4, 3q/4) are decoded as 1, others as 0
dec_bin = np.zeros_like(coeffs, dtype=np.int32)
for i in range(len(coeffs)):
    # If coefficient is in middle region, decode as 1
    if q/4 <= coeffs[i] < 3*q/4:
        dec_bin[i] = 1
    # Otherwise decode as 0

# Check if decryption was successful
decryption_success = np.array_equal(dec_bin[:len(m_bin)], m_bin[:len(dec_bin)])

# For VERBOSE output
if VERBOSE:
    print(f"I=\n{I}")
    print(f"Parameters: n={n}, q={q}, sigma={sigma}")
    print(f"s (secret key) has coefficients in range: [{min(s.coeffs)}, {max(s.coeffs)}]")
    print(f"e (error) has coefficients in range: [{min(e.coeffs)}, {max(e.coeffs)}]")
    print(f"a (public key component) is a random polynomial mod q")
    print(f"b (public key component) = a*s + e (mod q, x^n+1)")
    print(f"Original message length: {len(m)} characters, {len(m_bin)} bits")
    print(f"r has coefficients in range: [{min(r.coeffs)}, {max(r.coeffs)}]")
    print(f"e1 has coefficients in range: [{min(e1.coeffs)}, {max(e1.coeffs)}]")
    print(f"e2 has coefficients in range: [{min(e2.coeffs)}, {max(e2.coeffs)}]")
    print(f"Error term analysis:")
    _, er = np.polydiv(e * r, I)
    er = np.poly1d(np.mod(er.coeffs, q))
    _, e1s = np.polydiv(e1 * s, I)
    e1s = np.poly1d(np.mod(e1s.coeffs, q))
    error_term = np.poly1d(np.mod(er.coeffs + e2.coeffs - e1s.coeffs, q))
    print(f"Max coefficient in error term: {max(abs(error_term.coeffs))}")
    print(f"q/4 = {q/4} (error should be smaller than this)")

# Convert binary back to ASCII for verification
def bin_to_ascii(binary):
    result = ""
    # Process 8 bits at a time
    for i in range(0, len(binary), 8):
        if i + 8 <= len(binary):
            byte = binary[i:i+8]
            char_code = sum(bit << (7-j) for j, bit in enumerate(byte))
            result += chr(char_code)
    return result

original_text = bin_to_ascii(m_bin)
decrypted_text = bin_to_ascii(dec_bin[:len(m_bin) - len(m_bin) % 8])

if decryption_success:
    print("Decryption Successful!")
    if VERBOSE:
        print(f"Original text: {original_text}")
        print(f"Decrypted text: {decrypted_text}")
else:
    print("Decryption Failed!")
    if VERBOSE:
        print(f"Original bits: {m_bin[:32]}...")
        print(f"Decrypted bits: {dec_bin[:32]}...")
        # Calculate error rate
        incorrect_bits = sum(1 for a, b in zip(m_bin, dec_bin) if a != b)
        error_rate = incorrect_bits / len(m_bin)
        print(f"Error rate: {error_rate:.2%}")
