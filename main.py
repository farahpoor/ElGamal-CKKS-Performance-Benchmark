import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Crypto.PublicKey import ElGamal
from Crypto.Random import get_random_bytes
from Crypto.Util.number import inverse
import tenseal as ts
import random
import sys

###############################################################################
#                              FILE PATH
###############################################################################
FILE_PATH = "/Users/alifarahpoor/Desktop/allCountries.txt"  # <-- Update if needed

###############################################################################
#                         LOAD GEO DATA + MEMORY USAGE
###############################################################################
def load_geonames_data(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            usecols=[9, 10],
            names=["latitude", "longitude"],
            dtype={"latitude": float, "longitude": float}
        )
        # Load up to 100,000 rows
        return df.dropna().values[:100000]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

###############################################################################
#                               ELGAMAL CODE
###############################################################################
class ElGamalEncryption:
    def __init__(self, key_size=1024):
        print("Generating ElGamal key...")
        self.key = ElGamal.generate(key_size, randfunc=get_random_bytes)
        print("ElGamal key generated.")
        # Convert key parameters to plain Python ints
        self.p = int(self.key.p)
        self.g = int(self.key.g)
        self.y = int(self.key.y)
        self.x = int(self.key.x)

    def encrypt(self, value):
        value = int(value)
        k = random.randint(1, self.p - 1)
        c1 = pow(self.g, k, self.p)
        c2 = (value * pow(self.y, k, self.p)) % self.p
        return (c1, c2)

    def decrypt(self, c1, c2):
        s = pow(c1, self.x, self.p)
        s_inv = inverse(s, self.p)
        return (c2 * s_inv) % self.p

    # Additive homomorphism (simulate plaintext addition)
    def add_encrypted(self, enc1, enc2):
        c1_new = (enc1[0] * enc2[0]) % self.p
        c2_new = (enc1[1] * enc2[1]) % self.p
        return (c1_new, c2_new)

def test_performance_elgamal(data, max_rows=100000):
    """Runs ElGamal tests on powers of 2 up to max_rows."""
    encryptor = ElGamalEncryption()
    results = []
    print("\n--- Testing ElGamal ---")
    i = 1
    while True:
        current_size = 2 ** i
        if current_size < max_rows:
            input_size = min(current_size, len(data))
            i += 1
        else:
            input_size = min(max_rows, len(data))

        print(f"\n--> Starting test for input size: {input_size}")
        start_time = time.time()
        mem_before = get_memory_usage()

        try:
            # Scale lat/long by 1,000,000 to preserve decimals
            enc_values = [
                encryptor.encrypt(v * 1_000_000)
                for pair in data[:input_size]
                for v in pair
            ]
            # Add them all up
            result = enc_values[0]
            for j in range(1, len(enc_values)):
                result = encryptor.add_encrypted(result, enc_values[j])

            # Decrypt final sum just to verify
            _ = encryptor.decrypt(result[0], result[1])

            elapsed_time = time.time() - start_time
            mem_after = get_memory_usage()
            print(f"--> Finished step. Time: {elapsed_time:.4f}s "
                  f"| Memory used: {mem_after - mem_before:.2f} MB")
            results.append((input_size, elapsed_time, mem_after - mem_before))
        except Exception as e:
            print(f"FAILED at Input Size {input_size}: {e}")
            break

        if input_size == min(max_rows, len(data)):
            break

    return results

###############################################################################
#                CKKS CODE (EXACT SNIPPET WITH MINIMAL CHANGES)
###############################################################################
class TenSEAL_CKKS_Encryption:
    def __init__(
        self,
        poly_modulus_degree=16384,
        # 12 primes total => sum of bit sizes = 420 < 438
        coeff_mod_bit_sizes=[
            60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60
        ],
        global_scale=2**20
    ):
        # EXACT code from your snippet, minimal changes
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree,
            0,  # plain_modulus=0 for CKKS
            coeff_mod_bit_sizes,
            ts.ENCRYPTION_TYPE.ASYMMETRIC,
            1  # single-thread
        )
        self.context.global_scale = global_scale
        self.context.generate_galois_keys()
        
    def encrypt(self, value):
        return ts.ckks_vector(self.context, [value])
    
    def decrypt(self, enc_vector):
        return enc_vector.decrypt()[0]
    
    def multiply_encrypted(self, enc1, enc2):
        return enc1 * enc2

def tree_multiply(cipher_list, encryptor):
    # Recursively pair up ciphertexts and multiply them
    if len(cipher_list) == 1:
        return cipher_list[0]
    new_list = []
    for i in range(0, len(cipher_list), 2):
        if i + 1 < len(cipher_list):
            new_list.append(
                encryptor.multiply_encrypted(cipher_list[i], cipher_list[i+1])
            )
        else:
            new_list.append(cipher_list[i])
    return tree_multiply(new_list, encryptor)

def test_performance_tenseal(data, max_rows=100000):
    """Runs the provided TenSEAL CKKS code on powers of 2 up to max_rows."""
    encryptor = TenSEAL_CKKS_Encryption()
    results = []
    print("\n--- Testing TenSEAL CKKS ---")
    i = 1
    while True:
        current_size = 2 ** i
        if current_size < max_rows:
            input_size = min(current_size, len(data))
            i += 1
        else:
            input_size = min(max_rows, len(data))

        print(f"\n--> Starting test for input size: {input_size}")
        start_time = time.time()
        mem_before = get_memory_usage()

        try:
            # Encrypt each (latitude, longitude) in the data slice
            enc_values = [encryptor.encrypt(v) for pair in data[:input_size] for v in pair]
            
            # Multiply them all
            result = tree_multiply(enc_values, encryptor)
            
            # Decrypt final result to confirm success
            _ = encryptor.decrypt(result)
            
            elapsed_time = time.time() - start_time
            mem_after = get_memory_usage()
            print(f"--> Finished step. Time: {elapsed_time:.4f}s "
                  f"| Memory used: {mem_after - mem_before:.2f} MB")
            results.append((input_size, elapsed_time, mem_after - mem_before))
        except Exception as e:
            print(f"FAILED at Input Size {input_size}: {e}")
            break

        if input_size == min(max_rows, len(data)):
            break

    return results

###############################################################################
#     PLOT FUNCTIONS: SINGLE-SCHEME & COMBINED
###############################################################################
def plot_results_elgamal(elgamal_results):
    """Simple plot for ElGamal alone, if desired."""
    if not elgamal_results:
        print("No ElGamal results to plot.")
        return
    sizes, times, mems = zip(*elgamal_results)

    plt.figure(figsize=(12, 5))
    # Execution Time
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o', label="ElGamal")
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (s)")
    plt.title("ElGamal Execution Time")
    plt.legend()

    # Memory Usage
    plt.subplot(1, 2, 2)
    plt.plot(sizes, mems, marker='s', label="ElGamal")
    plt.xlabel("Input Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("ElGamal Memory Usage")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_results_tenseal(tenseal_results):
    """Exact snippet's plot function for CKKS alone."""
    if not tenseal_results:
        print("No results to plot.")
        return

    sizes, times, mems = zip(*tenseal_results)

    plt.figure(figsize=(12, 5))
    # Execution Time
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o', label="TenSEAL CKKS")
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time")
    plt.legend()

    # Memory Usage
    plt.subplot(1, 2, 2)
    plt.plot(sizes, mems, marker='s', label="TenSEAL CKKS")
    plt.xlabel("Input Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_results_combined(elgamal_results, ckks_results):
    """Combined plot with two lines: ElGamal + TenSEAL CKKS."""
    plt.figure(figsize=(12, 5))

    # Unpack ElGamal
    if elgamal_results:
        eg_sizes, eg_times, eg_mems = zip(*elgamal_results)
    else:
        eg_sizes, eg_times, eg_mems = [], [], []

    # Unpack CKKS
    if ckks_results:
        ck_sizes, ck_times, ck_mems = zip(*ckks_results)
    else:
        ck_sizes, ck_times, ck_mems = [], [], []

    # Subplot (1) Execution Time
    plt.subplot(1, 2, 1)
    if eg_sizes:
        plt.plot(eg_sizes, eg_times, marker='o', label="ElGamal")
    if ck_sizes:
        plt.plot(ck_sizes, ck_times, marker='s', label="TenSEAL CKKS")
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison")
    plt.legend()

    # Subplot (2) Memory Usage
    plt.subplot(1, 2, 2)
    if eg_sizes:
        plt.plot(eg_sizes, eg_mems, marker='o', label="ElGamal")
    if ck_sizes:
        plt.plot(ck_sizes, ck_mems, marker='s', label="TenSEAL CKKS")
    plt.xlabel("Input Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":
    data = load_geonames_data(FILE_PATH)
    if data is None:
        print("Failed to load dataset. Exiting.")
        sys.exit()

    print("Choose method to test:")
    print("1 - ElGamal only")
    print("2 - TenSEAL CKKS only")
    print("3 - Both")
    method = input("Enter choice (1/2/3): ")

    if method == "1":
        # ElGamal only
        elgamal_results = test_performance_elgamal(data, max_rows=100000)
        plot_results_elgamal(elgamal_results)

    elif method == "2":
        # TenSEAL CKKS only
        ckks_results = test_performance_tenseal(data, max_rows=100000)
        plot_results_tenseal(ckks_results)

    elif method == "3":
        # Both
        elgamal_results = test_performance_elgamal(data, max_rows=100000)
        ckks_results = test_performance_tenseal(data, max_rows=100000)
        plot_results_combined(elgamal_results, ckks_results)

    else:
        print("Invalid choice.")
