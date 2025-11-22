import sys
import os
import json
import heapq
import pickle
from collections import Counter
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    """Node in the Huffman tree"""
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoding:
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}
        self.root = None

    def build_frequency_table(self, tokens):
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return freq

    def build_huffman_tree(self, freq_table):
        heap = [Node(char=char, freq=freq) for char, freq in freq_table.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = Node(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)
        return heap[0]


    def generate_codes(self, node, prefix="", output_file=None, freq_table=None):
        if node is None:
            return
    
        if node.char is not None:
            code = prefix if prefix else "0"
            self.codes[node.char] = code
            self.reverse_codes[code] = node.char
    
            # Write to file if path provided
            if output_file:
                with open(output_file, "a", encoding="utf-8") as f:
                    char_repr = node.char if node.char not in ['\n', '\t', ' '] else repr(node.char)
                    freq = freq_table[node.char] if freq_table else "?"
                    bits_used = freq * len(code)
                    f.write(f"{char_repr:<10} | {freq:<10} | {code:<10} | {bits_used}\n")
            return
    
        self.generate_codes(node.left, prefix + "0", output_file, freq_table)
        self.generate_codes(node.right, prefix + "1", output_file, freq_table)

        
    def encode(self, text, code_output_path=None, mode="char"):
        """
        Compress text using Huffman codes.
        mode: "char" for character-level (default), "word" for word-level compression.
        """
        if not text:
            return "", {}
    
        # Step 1 — Tokenize input based on mode
        if mode == "word":
            tokens = text.split()          # Split by whitespace into words
        else:
            tokens = list(text)            # Split into characters
    
        # Step 2 — Build frequency table and tree
        freq_table = self.build_frequency_table(tokens)
        self.root = self.build_huffman_tree(freq_table)
    
        # Step 3 — Prepare output file
        if code_output_path and os.path.exists(code_output_path):
            os.remove(code_output_path)
    
        # Step 4 — Write header info
        if code_output_path:
            with open(code_output_path, "w", encoding="utf-8") as f:
                level = "Word-Level" if mode == "word" else "Character-Level"
                f.write(f"Huffman Encoding Report ({level})\n")
                f.write(f"Original Text Length: {len(tokens)} tokens\n")
                f.write("Symbol Table:\n")
                f.write("Symbol | Frequency | Huffman Code | Bits Used\n")
                f.write("=" * 60 + "\n")
    
        # Step 5 — Generate codes & optionally write them
        self.generate_codes(self.root, output_file=code_output_path, freq_table=freq_table)
    
        # Step 6 — Compute compression statistics
        compressed_bits = sum(freq_table[sym] * len(self.codes[sym]) for sym in freq_table)
        original_bits = len(tokens) * (8 if mode == "char" else 16)  # Approx per-symbol bits
        compression_ratio = (1 - compressed_bits / original_bits) * 100
    
        # Step 7 — Write summary
        if code_output_path:
            with open(code_output_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Original size (bits): {original_bits}\n")
                f.write(f"Compressed size (bits): {compressed_bits}\n")
                f.write(f"Compression ratio: {compression_ratio:.2f}%\n")
                f.write(f"Unique symbols: {len(freq_table)}\n")
                most_freq = max(freq_table, key=freq_table.get)
                least_freq = min(freq_table, key=freq_table.get)
                f.write(f"Most frequent symbol: '{most_freq}' ({freq_table[most_freq]} times)\n")
                f.write(f"Least frequent symbol: '{least_freq}' ({freq_table[least_freq]} times)\n")
    
        # Step 8 — Build final encoded string
        encoded = "".join(self.codes[sym] for sym in tokens)
        return encoded, freq_table

    
    def decode(self, encoded_text, freq_table, mode="char"):
        """Decode an encoded Huffman string using the frequency table."""
        if not encoded_text or not freq_table:
            return ""
    
        # Rebuild the Huffman tree
        self.root = self.build_huffman_tree(freq_table)
        self.generate_codes(self.root)
    
        decoded = []
        node = self.root
    
        for bit in encoded_text:
            # Defensive: skip invalid bits
            if node is None:
                node = self.root
                continue
    
            # Traverse tree
            node = node.left if bit == "0" else node.right
    
            # Leaf reached
            if node and node.char is not None:
                decoded.append(node.char)
                node = self.root
    
        return " ".join(decoded) if mode == "word" else "".join(decoded)

    
    def calculate_compression_ratio(self, original_size_bits, compressed_size_bits):
        """Return compression ratio as percentage saved.
        original_size_bits and compressed_size_bits are integers (bits)."""
        if original_size_bits <= 0:
            return 0.0
        return (1 - (compressed_size_bits / original_size_bits)) * 100
        

    def visualize_tree(self, root, filename="huffman_tree"):
        if not root:
            print("[WARNING] No Huffman tree found.")
            return
    
        dot = Digraph(comment="Huffman Binary Tree", format="png")
        dot.attr('node', shape='circle', style='filled', color='lightblue2', fontname="Helvetica")
    
        def add_nodes_edges(node, parent=None, edge_label=""):
            if not node:
                return
            node_label = f"{node.char if node.char else '*'}\n{node.freq}"
            dot.node(str(id(node)), node_label)
            if parent:
                dot.edge(str(id(parent)), str(id(node)), label=edge_label, color='red')
            add_nodes_edges(node.left, node, "0")
            add_nodes_edges(node.right, node, "1")
    
        add_nodes_edges(root)
        dot.attr(label="Huffman Binary Tree", fontsize="20", labelloc="t", fontname="Helvetica-Bold")
        output_path = dot.render(filename, cleanup=True)
        print(f"[INFO] Huffman tree saved as {output_path}")

    def pack_bits(self, bitstring):
        """Convert a string like '01001101...' into real bytes."""
        # Pad bitstring to make it divisible by 8
        padding = 8 - (len(bitstring) % 8)
        if padding != 8:
            bitstring += "0" * padding
    
        # Save padding info (needed during decode)
        return int(bitstring, 2).to_bytes(len(bitstring) // 8, byteorder='big'), padding

    def unpack_bits(self, byte_data, padding):
        bitstring = ''.join(f"{byte:08b}" for byte in byte_data)
        if padding:
            bitstring = bitstring[:-padding]
        return bitstring


# Helper functions for encoding to binary
def pad_encoded_text(encoded_text):
    extra_padding = 8 - len(encoded_text) % 8
    encoded_text += "0" * extra_padding
    padded_info = "{0:08b}".format(extra_padding)
    return padded_info + encoded_text


def remove_padding(padded_encoded_text):
    padded_info = padded_encoded_text[:8]
    extra_padding = int(padded_info, 2)
    encoded_text = padded_encoded_text[8:]
    return encoded_text[:-extra_padding]


def to_bytes(padded_encoded_text):
    b = bytearray()
    for i in range(0, len(padded_encoded_text), 8):
        byte = padded_encoded_text[i:i + 8]
        b.append(int(byte, 2))
    return bytes(b)



def compress_file(file_path, mode="char"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    hc = HuffmanCoding()

    encoded, freq_table = hc.encode(
        text, 
        code_output_path=f"{file_path}_codes.txt",
        mode=mode
    )

    # Pack into real binary bytes
    packed_bytes, padding = hc.pack_bits(encoded)

    #compressed_path = f"compress_{os.path.basename(file_path)}.huff"
    compressed_path = f"{os.path.basename(file_path)}.huff"
    with open(compressed_path, "wb") as f:
        f.write(bytes([padding]))      # first byte = padding length
        f.write(packed_bytes)          # remaining bytes = packed bits

    # Save the frequency table
    with open(compressed_path + ".freq.json", "w", encoding="utf-8") as f:
        json.dump(freq_table, f, indent=2)

    # Tree visualization
    file_nm = os.path.basename(file_path) 
    tree_file_nm = file_nm + "_tree.png"
    if hasattr(hc, "root") and hc.root:
        hc.visualize_tree(hc.root, filename=tree_file_nm)

    print(f"[INFO] Compression complete using {mode}-level Huffman coding.")
    print(f"[INFO] Output: {compressed_path}")


def decompress_file(file_path, mode="char"):
    hc = HuffmanCoding()

    # Read compressed binary
    with open(file_path, "rb") as f:
        padding = f.read(1)[0]       # first byte = padding
        packed_data = f.read()       # the rest = packed Huffman bits

    # Load frequency table
    freq_path = file_path + ".freq.json"
    with open(freq_path, "r", encoding="utf-8") as f:
        freq_table = json.load(f)

    # Unpack bits
    encoded = hc.unpack_bits(packed_data, padding)

    # Decode to original data
    decoded = hc.decode(encoded, freq_table, mode=mode)

    output_path = "uncompressed_" + os.path.basename(file_path).replace(".huff", ".txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(decoded)

    print(f"[INFO] Decompression complete using", mode, "mode.")
    print(f"[INFO] Output:", output_path)



# Assuming you already have compress_file() and decompress_file() defined
# and both functions accept a `mode` argument (default "char")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python huffman_tool.py <file_path> <compress|decompress> [char|word]")
        sys.exit(1)

    file_path = sys.argv[1]
    operation = sys.argv[2].lower()
    mode = sys.argv[3].lower() if len(sys.argv) == 4 else "char"

    if mode not in ("char", "word"):
        print("Invalid encoding mode. Use 'char' or 'word'.")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    if operation == "compress":
        compress_file(file_path, mode=mode)
    elif operation == "decompress":
        decompress_file(file_path, mode=mode)
    else:
        print("Invalid operation. Use 'compress' or 'decompress'.")
        sys.exit(1)

    