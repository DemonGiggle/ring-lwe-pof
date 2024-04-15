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

VERBOSE=0

#
# Define a quotient ring as R_q = Z_q[x]/I
#

# The order of the quotient ring is defined as:
# n should be power of 2
n = pow(2, 9)

# The ideal in the quotient ring is defined as below
# I = <x^n+1>
I = np.poly1d( [1] + [0]*(n-1) + [1] )

# q should be a prime and q = 1 mod 2n
# but this prime is hard to find, so we relax the restriction
# to only a big prime. This relaxion might be conflict with
# the security hardness proof.
q = 2443041973

# Now we generate a keypair
#
# First, we start with secret key s <- R_q. Note that the
# secret key should be small enough with enlarge the error term.
#
# ref. when we decode the message, there is an error term like this:
# (r*e - s*e1 + e2), if s is large, these error term will be large to
# affect the final decoding result.
# Here, for simplicty, we just use either 0 or 1 for the secret key.
# Of course, this will have huge impact on security.
#
s = np.poly1d(np.random.randint(0, 2, n))

# Generate an error e <- R_q
# Note that the error should be small enough. Limit the error to {0, 1}
e = np.poly1d(np.random.randint(0, 2, n))

# The public key pair is a tuple (a, b), where
# a <- R_q choosing ranomly, and b = a * s + e (mod q)
a = np.poly1d(np.random.randint(0, q, n))

_, b = np.polydiv(a * s, I)
b = np.poly1d( np.mod(b + e, q) )

#
# Start to encrypt and decrypt a message
#

# To avoid padding, we create a message just fit into n-bit
m = ''
for i in range(int(n / 8)):
    m = m + chr(ord('a')+i%26)
    
# Convert message into binary array and then to polynomial
m_bin = [int(bit) for char in m for bit in format(ord(char), '08b')]
m_poly = np.poly1d(m_bin)

# Now, we start to encrypt the message m to (u, v) <- R_q
# where, 
#   u = a * r + e1 (mod q)
#   v = b * r + e2 + [q/2] * m (mod q)
#
# From above, we need to sample r, e1, e2 from R_q. Note that
# these three samples should be small enough without affecting
# the final result. How small? Just give them either 0 or 1 should
# be small enough but might have big security concern.
#
r = np.poly1d(np.random.randint(0, 2, n))
e1 = np.poly1d(np.random.randint(0, 2, n))
e2 = np.poly1d(np.random.randint(0, 2, n))

_, u = np.polydiv(a * r, I)
u = np.poly1d(np.mod(u + e1, q))

_, v = np.polydiv(b * r, I)
v = np.poly1d(np.mod(v + e2 + q/2 * m_poly, q))

# Now, let's decode it.
#
# The decode fomular is:
#
#   v - u * s = (r * e - s * e1 + e2) + [q/2] * m   (mod q)
#
# Here, the error term (r * e - s * e1 + e2) is smaller than q/4. Why?
# Because the coefficients within r, e, s, e1, e2 are all either 0 or 1.
#
# We choose any two of them and perform polynomial multiplication, the
# coefficients of final polynomail is at most 2(n-1). Here is why.
#
# Without lost of generality, we choose r, e which are in R_q and there
# coefficients are either 0 or 1 as stated before.
#
# let a_(n-1) denotes the coefficient from x^(n-1) in r, and
# b_(n-1) the coefficient from x^(n-1) in e.
#
# d = r * e will have a coefficient d_k = a_0*b_k + a_1*b_(k-1) + ... + a_k*b_0
# The maximum coefficient occurs when k = n, which has the most items to contribute to it.
#
# So d_(n-1) has n terms and each terms at most 1, so maximum of d_(n-1) is (n-1).
# 
# After dividing the I, at most one term are wrap back to d_(n-1), this adds the coefficient d_(n-1)
# with one more (n-1). That means, d_(n-1) is at most 2(n-1).
#
# By design, q is large enough than n, so q/4 should be also large than 2(n-1). It is safe that
#
# The error term above is neglible when we decode the message.
#
_, us = np.polydiv(u * s, I)
us = np.poly1d(np.mod(us, q))
d_poly = np.poly1d(np.mod(v - us, q))

# Now check each coefficient to see whether they are near 0 or q/2
# Output 0 if it is near 0, otherwise output 1
coeffs = d_poly.coeffs

# poly1d.coeffs will trim the leading zeros
coeffs = np.concatenate( (np.zeros(n - coeffs.size), coeffs) )

# The numbers are within [0, q) and it could be divided to 3 sections:
# [0, q/4), [q/4, 3*q/4), [3*q/4, q)
#
# From the equation
#
#   v - u * s = (r * e - s * e1 + e2) + [q/2] * m   (mod q)
#
# We could clearly see that if the error terms (r*e-s*e1+e2) is small
# enough (within q/4), then it could be considered as a perturbation
# around q/2 if m_i is 1, or around 0 or q if m_i is 0.
#
# Note that the order of the following operations is critical. We have
# to calculate coeffs which will be clampped to 0, and then those to 1.
#
coeffs = np.where((coeffs < q/4) | ((coeffs >= 3*q/4) & (coeffs < q)), 0, coeffs)
coeffs = np.where(coeffs > 1, 1, coeffs) # since the remaining numbers other than 0 is clamped to 1 
dec_bin = coeffs.astype(np.int32)


if VERBOSE:
    print(f"I=\n{I}")
    print(f"s=\n{s}")
    print(f"e=\n{e}")
    print(f"a=\n{a}")
    print(f"b=\n{b}")
    print(f"m={m}\nm_bin={m_bin}\nm_poly=\n{m_poly}")
    print(f"r=\n{r}")
    print(f"e1=\n{e1}")
    print(f"e2=\n{e2}")
    print(f"u=\n{u}")
    print(f"v=\n{v}")
    print(f"us=\n{us}")
    print(f"v-us=\n{d_poly}")
    print(dec_bin == m_bin)

if (dec_bin == m_bin).all():
    print("Success")
else:
    print("Fail!!")
