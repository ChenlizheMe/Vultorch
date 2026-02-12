"""Convert a SPIR-V binary (.spv) to a C header with a uint32_t array."""
import sys
import os

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input.spv> <output.h> <variable_name>")
        sys.exit(1)

    spv_path   = sys.argv[1]
    header_path = sys.argv[2]
    var_name    = sys.argv[3]

    with open(spv_path, 'rb') as f:
        data = f.read()

    # Pad to uint32_t alignment
    while len(data) % 4 != 0:
        data += b'\x00'

    words = []
    for i in range(0, len(data), 4):
        word = int.from_bytes(data[i:i+4], byteorder='little')
        words.append(f"0x{word:08x}")

    os.makedirs(os.path.dirname(header_path) or '.', exist_ok=True)

    with open(header_path, 'w') as f:
        f.write(f"// Auto-generated from {os.path.basename(spv_path)} — do not edit\n")
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"static const uint32_t {var_name}[] = {{\n")
        for i in range(0, len(words), 8):
            line = ", ".join(words[i:i+8])
            f.write(f"    {line},\n")
        f.write("};\n")
        f.write(f"static const size_t {var_name}_size = sizeof({var_name});\n")

if __name__ == '__main__':
    main()
