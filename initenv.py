from pathlib import Path

if Path('.env').is_file():
    print('.env already exists.')
    exit()

with Path('.env').open('w') as fp:
    print(f'WORKDIR={str(Path.cwd().resolve())}', file=fp)
    print(f'DATADIR={str(Path("data").resolve())}', file=fp)
