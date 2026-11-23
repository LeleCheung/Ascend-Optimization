"""Compare executable ELF contents while retaining differences in debug metadata."""
import hashlib
from pathlib import Path
import struct


def execution_signature(path):
    return execution_signature_bytes(Path(path).read_bytes())


def execution_signature_bytes(data):
    assert data[:6] == b'\x7fELF\x02\x01', 'Expected ELF64 little endian'
    header = struct.unpack_from('<HHIQQQIHHHHHH', data, 16)
    kind, machine, version, entry, _, section_offset, flags, _, _, _, size, count, names_index = header
    assert size == 64 and count > 0
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, section_offset+i*size) for i in range(count)]
    strings = sections[names_index]
    names = data[strings[4]:strings[4]+strings[5]]
    def name(table, offset):
        return table[offset:table.index(b'\0', offset)].decode('utf8')
    records, symbols = {}, []
    for section in sections:
        label = name(names, section[0])
        if section[2] & 2 or label in {'.ascend.stack.size.record', '__CCE_KernelArgSize'}:
            body = data[section[4]:section[4]+section[5]] if section[1] != 8 else b''
            records[label] = dict(type=section[1], flags=section[2], address=section[3],
                size=section[5], alignment=section[8], sha256=hashlib.sha256(body).hexdigest())
        if section[1] == 2:
            table = sections[section[6]]
            table_data = data[table[4]:table[4]+table[5]]
            assert section[9] == 24
            for offset in range(section[4], section[4]+section[5], 24):
                label_offset, info, other, index, value, symbol_size = struct.unpack_from('<IBBHQQ', data, offset)
                if info >> 4 in {1, 2}:
                    symbols.append(dict(name=name(table_data, label_offset), info=info, other=other,
                        section=name(names, sections[index][0]) if 0 < index < count else index,
                        value=value, size=symbol_size))
    assert '.text' in records and symbols
    return dict(header=dict(type=kind, machine=machine, version=version, entry=entry, flags=flags),
                sections=records, public_symbols=sorted(symbols, key=lambda x: x['name']))
