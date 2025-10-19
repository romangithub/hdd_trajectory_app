# hdd_toolkit.spec — универсальная спецификация для Windows/macOS
# Собирает один каталог (папка с exe/.app и всем нужным)

import sys
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
# Matplotlib и его бэкенды часто требуют явных импортов
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('matplotlib.backends')
hiddenimports += collect_submodules('pandas')

block_cipher = None

datas = [
    ('Home.py', '.'),
    ('.streamlit', '.streamlit'),
    ('modules', 'modules'),
    ('pages', 'pages'),
]

a = Analysis(
    ['app_launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=['rthooks/set_cwd.py'],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# console=False — без чёрного окна; для отладки можно поставить True
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HDD Toolkit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # <-- окно консоли не показывать
    icon=None        # можешь указать .ico/.icns
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    name='HDD Toolkit'
)
