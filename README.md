# ElGamal and CKKS Homomorphic Encryption Comparison

This repository provides a Python-based benchmark suite that compares **ElGamal** and **CKKS** (via [TenSEAL](https://github.com/OpenMined/TenSEAL)) homomorphic encryption. It measures both **execution time** and **memory usage** for encrypting and combining large datasets of latitude-longitude coordinates.

## Features
- **ElGamal Encryption** (using [PyCryptodome](https://pycryptodome.readthedocs.io/en/latest/)).
- **CKKS** (using [TenSEAL](https://github.com/OpenMined/TenSEAL)) for efficient floating-point homomorphic operations.
- **Command-line prompt** to run either:
  1. ElGamal only
  2. TenSEAL CKKS only
  3. Both, side by side

- **Performance metrics**:
  - Execution time (seconds)
  - Memory usage (MB)

- **Plots** comparing time and memory usage vs. input size.

## Requirements
1. **Python 3.8+** (tested with Python 3.9).
2. [pip](https://pypi.org/project/pip/) or another package manager.
3. Basic build tools (often required for TenSEAL).

## Installation
1. **Clone this repository** (or download the ZIP):

   ```bash
   git clone https://github.com/your-username/ElGamal-CKKS-Homomorphic-Comparison.git
   cd ElGamal-CKKS-Homomorphic-Comparison
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux

   # or on Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   psutil
   pycryptodome
   numpy
   pandas
   matplotlib
   tenseal
   ```

## Usage
1. **Update the file path** to your dataset in the script:

   ```python
   FILE_PATH = "/path/to/allCountries.txt"
   ```

   The file should be a TSV with latitude and longitude in columns 9 and 10.

2. **Run the script**:

   ```bash
   python main.py
   ```

3. **Choose a test method when prompted**:

   ```
   Choose method to test:
   1 - ElGamal only
   2 - TenSEAL CKKS only
   3 - Both
   ```

4. **Results** will be printed in the terminal and shown in a comparison plot.

## Common Issues

- **"Failed to find enough qualifying primes"**  
  TenSEAL could not find a suitable set of primes for the given parameters. Adjust `coeff_mod_bit_sizes` or reduce `poly_modulus_degree`.

- **"End of modulus switching chain reached"**  
  You ran out of multiplicative depth. Add more primes (levels), reduce the global scale, or limit the number of multiplications.

## License

MIT License
