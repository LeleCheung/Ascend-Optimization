import json
import shutil
from pathlib import Path

path = Path('/root/.claude/settings.json')
settings = json.loads(path.read_text(encoding='utf-8'))
updates = {}

if settings.get('model') == 'deepseek-flash[1M]':
    settings['model'] = 'deepseek-flash'
    updates['model'] = 'deepseek-flash'

for key, value in settings.get('env', {}).items():
    if ('MODEL' in key) and value == 'deepseek-flash[1M]':
        settings['env'][key] = 'deepseek-flash'
        updates[f'env.{key}'] = 'deepseek-flash'

if updates:
    backup = path.with_name('settings.json.pre-kernelgen-model-fix')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

print(json.dumps({'updated': updates, 'backup': str(backup) if updates else None}))
