"""Read queue inputs from the ZIP or an atomically verified extraction."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile


class DirectoryInputs:
    def __init__(self, directory, receipt):
        self.directory = Path(directory).resolve()
        if receipt.get('complete') is not True or not receipt.get('files'):
            raise ValueError('Extraction receipt is incomplete')
        self.entries = {}
        for item in receipt['files']:
            name = item['name']
            if Path(name).name != name or not name.lower().endswith('.pdf') or name in self.entries:
                raise ValueError('Unsafe or duplicate input name')
            self.entries[name] = SimpleNamespace(filename=name, file_size=int(item['size']), is_dir=lambda: False)

    def infolist(self):
        return list(self.entries.values())

    def open(self, name):
        info = self.entries[name]
        path = self.directory / name
        if path.is_symlink() or path.stat().st_size != info.file_size:
            raise ValueError('Extracted input changed: ' + name)
        return path.open('rb')


@contextmanager
def queue_source(archive, directory, receipt_path):
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        yield DirectoryInputs(directory, json.loads(receipt_path.read_text()))
    else:
        with zipfile.ZipFile(archive) as source:
            yield source
