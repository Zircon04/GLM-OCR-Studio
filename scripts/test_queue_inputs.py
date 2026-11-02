import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from queue_inputs import queue_source


class QueueInputsTests(unittest.TestCase):
    def test_folder_mode_does_not_need_zip(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); cache = root/'input-cache'; cache.mkdir()
            (cache/'тест.pdf').write_bytes(b'PDF unit fixture')
            receipt = root/'inputs-ready.json'
            receipt.write_text(json.dumps({'complete': True, 'files': [{'name': 'тест.pdf', 'size': 16}]}))
            with queue_source(root/'removed.zip', cache, receipt) as src:
                entries = src.infolist()
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0].file_size, 16)
                with src.open('тест.pdf') as f:
                    self.assertEqual(f.read(), b'PDF unit fixture')

    def test_zip_mode_remains_available(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); archive=root/'original.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('test.pdf', b'fixture')
            with queue_source(archive,root,root/'absent.json') as src:
                self.assertEqual(src.infolist()[0].filename,'test.pdf')

    def test_rejects_unsafe_receipt_member(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); receipt=root/'inputs-ready.json'
            receipt.write_text(json.dumps({'complete':True,'files':[{'name':'../escape.pdf','size':1}]}))
            with self.assertRaises(ValueError):
                with queue_source(root/'removed.zip',root,receipt):pass

    def test_rejects_incomplete_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); receipt=root/'inputs-ready.json'
            receipt.write_text(json.dumps({'complete':False,'files':[]}))
            with self.assertRaises(ValueError):
                with queue_source(root/'removed.zip',root,receipt):pass


if __name__ == '__main__':
    unittest.main()
