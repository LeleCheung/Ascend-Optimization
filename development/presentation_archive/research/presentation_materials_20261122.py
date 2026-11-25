"""Route presentation assets into the organized 06_work/material tree."""
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from xml.etree import ElementTree
from zipfile import ZipFile


def material_folder(filename: str) -> str:
    if filename.startswith('HG-KDA-overview-初稿') or filename == '开场与前四页讲稿.md':
        return 'archive/20261122_旧版'
    if filename in {f'HG-KDA-过渡与流程-三页-{i}.png' for i in (1, 2, 3)}:
        return 'archive/重复预览'
    if filename in ('开场与前五页讲稿.md', '指标比较-讲稿.md'):
        return '05_讲稿'
    if filename.startswith(('算子优化-基本回路', 'HG-KDA-overview-简版', 'HG-KDA-过渡与流程-三页')):
        return '01_开场与总览'
    if filename.startswith('指标比较-'):
        return '02_指标对比'
    if filename.startswith('模拟器-证据与优化'):
        return '03_模拟器'
    if filename.startswith('GQA-'):
        return '04_算子案例/GQA'
    raise ValueError(f'Unclassified presentation asset: {filename}')


def material_target(delivery: Path, filename: str) -> Path:
    delivery = delivery.resolve()
    if delivery.name == '06_work':
        return delivery / 'material' / material_folder(filename) / filename
    return delivery / filename


def copy_material(source: Path, delivery: Path) -> Path:
    target = material_target(delivery, source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if source.suffix.lower() == '.xlsx' and source.parent.resolve() != target.parent.resolve():
        repair_workbook_links(source, target)
    return target


def repair_workbook_links(source: Path, target: Path) -> list[dict]:
    """Keep source-relative evidence hyperlinks usable after workbook delivery."""
    changes = []
    replacements = {}
    with ZipFile(target) as book:
        for name in book.namelist():
            if not name.endswith('.rels'):
                continue
            root = ElementTree.fromstring(book.read(name))
            changed = False
            for relationship in root:
                link = relationship.get('Target', '')
                if not relationship.get('Type', '').endswith('/hyperlink') or urlsplit(link).scheme:
                    continue
                original = (source.parent / Path(link.replace('\\', '/'))).resolve()
                if original.exists():
                    relationship.set('Target', original.as_uri())
                    changes.append({'previous': link, 'target': str(original)})
                    changed = True
            if changed:
                replacements[name] = ElementTree.tostring(root, encoding='utf-8', xml_declaration=True)
        if not replacements:
            return changes
        temporary = target.with_name(target.name + '.relocation-tmp')
        with ZipFile(temporary, 'w') as output:
            for entry in book.infolist():
                output.writestr(entry, replacements.get(entry.filename, book.read(entry.filename)))
    temporary.replace(target)
    return changes
