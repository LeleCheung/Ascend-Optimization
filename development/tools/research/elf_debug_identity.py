"""Verify ELF execution content while allowing only debug location changes."""
import hashlib
from pathlib import Path
import struct


def debug_section(name):
    return name.startswith(('.debug_', '.zdebug_', '.rela.debug_', '.rel.debug_'))


def elf(path):
    raw = Path(path).read_bytes()
    assert raw[:6] == b'\x7fELF\x02\x01', 'Expected little-endian ELF64'
    header = struct.unpack_from('<HHIQQQIHHHHHH', raw, 16)
    assert header[0] in {1, 2}, 'Expected relocatable or executable ELF'
    segments = []
    for i in range(header[9]):
        row = struct.unpack_from('<IIQQQQQQ', raw, header[4] + i * header[8])
        segments.append(dict(header=row, data=raw[row[2]:row[2] + row[5]]))
    offset, size, count, strings = header[5], header[10], header[11], header[12]
    rows = [struct.unpack_from('<IIQQQQIIQQ', raw, offset + i * size) for i in range(count)]
    table = rows[strings]
    names = raw[table[4]:table[4] + table[5]]
    sections = {}
    order = []
    for row in rows:
        name = names[row[0]:].split(b'\0')[0].decode()
        assert name not in sections
        order.append(name)
        sections[name] = dict(type=row[1], flags=row[2], address=row[3], size=row[5],
            link=row[6], info=row[7], alignment=row[8], entry_size=row[9],
            data=raw[row[4]:row[4] + row[5]] if row[1] != 8 else b'')
    return dict(header=(raw[:16], tuple(header[i] for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])),
                sections=sections, order=order, segments=segments)


def verify(original, observed):
    a, b = elf(original), elf(observed)
    assert a['header'] == b['header'] and a['order'] == b['order'], 'ELF architecture or section set differs'
    assert a['segments'] == b['segments'], 'Program headers or loaded segment bytes differ'
    changed = []
    allocated = {}
    symbol_debug_changes = 0
    for name, old in a['sections'].items():
        new = b['sections'][name]
        if old != new:
            changed.append(name)
        if debug_section(name):
            assert not (old['flags'] & 2 or new['flags'] & 2), 'Debug section unexpectedly allocated'
            continue
        if old['type'] == 2:
            assert {k: v for k, v in old.items() if k != 'data'} == {
                k: v for k, v in new.items() if k != 'data'}
            assert old['entry_size'] == 24
            for offset in range(0, old['size'], 24):
                x = struct.unpack_from('<IBBHQQ', old['data'], offset)
                y = struct.unpack_from('<IBBHQQ', new['data'], offset)
                if x != y:
                    assert (x[:4], x[5]) == (y[:4], y[5]), 'Non-location symbol attributes differ'
                    assert x[3] < len(a['order']) and debug_section(a['order'][x[3]]), 'Executable symbol differs'
                    symbol_debug_changes += 1
        else:
            assert old == new, 'Non-debug section differs: ' + name
        if old['flags'] & 2:
            allocated[name] = dict(size=old['size'], sha256=hashlib.sha256(old['data']).hexdigest())
    assert allocated, 'No allocated executable/data sections'
    return dict(status='same_executable_and_data_debug_locations_differ',
        original=str(original), original_sha256=hashlib.sha256(Path(original).read_bytes()).hexdigest(),
        observed=str(observed), observed_sha256=hashlib.sha256(Path(observed).read_bytes()).hexdigest(),
        allocated_sections=allocated, changed_sections=changed, debug_symbol_value_changes=symbol_debug_changes,
        scope='Program headers, loaded segments, all non-debug sections, executable/data bytes, relocations '
              'and non-debug symbol fields are identical. '
              'Only unallocated debug sections and values of symbols referring to debug sections may differ. '
              'Full binary hashes remain distinct. This comparison is used only for unchanged auxiliary kernels.')
