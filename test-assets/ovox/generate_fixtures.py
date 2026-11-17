import json, struct, zlib, lzma, os

OUT = r"G:/jobs/candle-based-projects/candle-3d/test-assets/ovox"

def expand(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

def z_order(x, y, z):
    return (expand(x) * 4 + expand(y) * 2 + expand(z)) & 0xFFFFFFFF

def encode_svo(codes, depth):
    svo = [0]
    stack = [0] * (depth - 1)
    insert_stack = [0] * depth
    stack_ptr = [0] * depth
    stack_ptr[0] = 0
    for i, code in enumerate(codes):
        for j in range(depth):
            insert_stack[j] = (code >> (3 * (depth - 1 - j))) & 0x7
        if i == 0:
            insert_from = 0
        else:
            insert_from = depth - 1
            for j in range(depth - 1):
                if insert_stack[j] != stack[j]:
                    insert_from = j
                    break
        for j in range(insert_from, depth):
            if j > insert_from:
                svo.append(0)
                stack_ptr[j] = len(svo) - 1
            svo[stack_ptr[j]] |= 1 << insert_stack[j]
            if j < depth - 1:
                stack[j] = insert_stack[j]
    return bytes(svo)

def compress(data, algo, level):
    if algo == 'none':
        return data
    if algo == 'deflate':
        c = zlib.compressobj(level, wbits=-15)
        return c.compress(data) + c.flush()
    if algo == 'lzma':
        c = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'preset': level}])
        return c.compress(data) + c.flush()
    raise ValueError(algo)

def write_vxz(path, coord, attr, chunk_size=4, compression='lzma', level=None,
              attr_interleave='as_is'):
    level = level if level is not None else {'none': 0, 'deflate': 9, 'lzma': 9}[compression]
    n = len(coord)
    depth = chunk_size.bit_length() - 1
    assert 1 << depth == chunk_size
    file_info = {
        'num_voxel': n,
        'chunk_size': chunk_size,
        'filter': 'none',
        'compression': compression,
        'compression_level': level,
        'raw_size': n * 4 + sum(len(v) * len(v[0]) for v in attr.values()),
        'compressed_size': 0,
        'compress_ratio': 0.0,
        'attr_interleave': attr_interleave,
        'attr': [[k, len(v[0])] for k, v in attr.items()],
        'chunks': [],
    }
    groups = {}
    for i, c in enumerate(coord):
        key = (c[0] // chunk_size, c[1] // chunk_size, c[2] // chunk_size)
        groups.setdefault(key, []).append(i)
    bin_data = b''
    for key in sorted(groups):
        rows = groups[key]
        local = [[coord[i][d] % chunk_size for d in range(3)] for i in rows]
        codes = [z_order(*c) for c in local]
        order = sorted(range(len(codes)), key=lambda i: codes[i])
        sorted_codes = [codes[i] for i in order]
        svo = encode_svo(sorted_codes, depth)
        svo_bytes = compress(svo, compression, level)
        attr_rows = {k: [v[rows[i]] for i in order] for k, v in attr.items()}
        chunk_info = {'idx': list(key)}
        chunk_data = b''
        chunk_info['svo'] = [len(chunk_data), len(svo_bytes)]
        chunk_data += svo_bytes
        if attr_interleave == 'none':
            for k, rows_v in attr_rows.items():
                for ch in range(len(rows_v[0])):
                    data = bytes(row[ch] for row in rows_v)
                    packed = compress(data, compression, level)
                    name = f'{k}_{ch}'
                    chunk_info[name] = [len(chunk_data), len(packed)]
                    chunk_data += packed
        elif attr_interleave == 'as_is':
            for k, rows_v in attr_rows.items():
                data = b''.join(bytes(row) for row in rows_v)
                packed = compress(data, compression, level)
                chunk_info[k] = [len(chunk_data), len(packed)]
                chunk_data += packed
        elif attr_interleave == 'all':
            data = b''.join(b''.join(bytes(attr_rows[k][i]) for k in attr_rows) for i in range(len(rows)))
            packed = compress(data, compression, level)
            chunk_info['attr'] = [len(chunk_data), len(packed)]
            chunk_data += packed
        chunk_info['ptr'] = [len(bin_data), len(chunk_data)]
        bin_data += chunk_data
        file_info['chunks'].append(chunk_info)
    file_info['compressed_size'] = len(bin_data)
    file_info['compress_ratio'] = file_info['raw_size'] / len(bin_data) if bin_data else 0.0
    structure = json.dumps(file_info).encode()
    header = b'VXZ\x00' + struct.pack('>I', len(structure) + 8)
    with open(path, 'wb') as f:
        f.write(header); f.write(structure); f.write(bin_data)
    return file_info

coord = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[2,1,0],[3,3,3],
         [4,4,4],[5,0,0],[0,5,0]]
base_color = [[(i*3+j) % 256 for j in range(3)] for i in range(len(coord))]
metallic = [[(i*7+3) % 256] for i in range(len(coord))]
alpha = [[255-i] for i in range(len(coord))]
attr = {'base_color': base_color, 'metallic': metallic, 'alpha': alpha}

fixtures = [
    ('official_lzma_as_is.vxz', 'lzma', 'as_is'),
    ('official_lzma_none_interleave.vxz', 'lzma', 'none'),
    ('official_lzma_all_interleave.vxz', 'lzma', 'all'),
    ('official_deflate_as_is.vxz', 'deflate', 'as_is'),
    ('official_none_as_is.vxz', 'none', 'as_is'),
]
for name, comp, inter in fixtures:
    info = write_vxz(os.path.join(OUT, name), coord, attr, 4, comp, None, inter)
    print(name, 'chunks', len(info['chunks']), 'size', info['compressed_size'])

# multi-LZMA2-chunk raw stream (3 MiB, exceeds the 2 MiB LZMA2 chunk limit)
pattern = bytes(((i * 37 + 11) & 0xFF) for i in range(3 * 1024 * 1024))
c = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9}])
stream = c.compress(pattern) + c.flush()
with open(os.path.join(OUT, 'lzma2_multichunk.bin'), 'wb') as f:
    f.write(stream)
print('lzma2_multichunk.bin', len(stream), 'control bytes prefixes')
# report the LZMA2 chunk starts by parsing framing
pos = 0
starts = []
while pos < len(stream):
    ctrl = stream[pos]; pos += 1
    if ctrl == 0:
        break
    starts.append((ctrl, pos - 1))
    if ctrl in (1, 2):
        size = (stream[pos] << 8 | stream[pos+1]) + 1; pos += 2 + size
    else:
        unpacked = ((ctrl & 0x1F) << 16 | stream[pos] << 8 | stream[pos+1]) + 1
        packed = (stream[pos+2] << 8 | stream[pos+3]) + 1
        pos += 4
        if ctrl >= 0xC0:
            pos += 1
        pos += packed
print('chunk starts', starts[:8], '... total', len(starts), 'parsed', pos == len(stream))
print('expected sha-ish', sum(pattern) & 0xFFFFFFFF)
