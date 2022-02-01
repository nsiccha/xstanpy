from xstanpy.mdpy import *
import sys

replacement_rules = dict()

for path in pathlib.Path('xstanpy').glob('*.py'):
    lines = File(path).lines
    name = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line: continue
        if len(line) < 5: continue
        if line[:3] == 'def':
            line = line[4:]
        elif line[:5] == 'class':
            line = line[6:]
            name = []
        else: continue
        pos = line.find('(')
        if pos == -1: pos = line.find(':')
        line = line[:pos]
        if not name: name = [line]
        else: name = [name[0], line]
        key = '`{}`'.format('.'.join(name))
        val = f'[{key}](../{path}#L{i})'
        replacement_rules[key] = val

file_paths = (
    list(pathlib.Path('examples/stan').glob('*.stan'))
    + list(pathlib.Path('examples').glob('*.py'))
    + list(pathlib.Path('examples').glob('*.md'))
)

for path in file_paths:
    spath = pathlib.Path(*path.parts[1:])
    key = f'`{spath}`'
    val = f'[{key}]({spath})'
    replacement_rules[key] = val

config_paths = pathlib.Path('examples').glob('figs/*/*')
for path in config_paths:
    pngs = list(path.glob('*.png'))
    print(path)
    if not pngs: continue
    fig = sorted(
        path.glob('*.png'),
        key=lambda path: path.stat().st_mtime
    )[-1]
    spath = pathlib.Path(*fig.parts[1:])
    key = f'`{path.parts[-1]}`'
    val = f'[{key}]({spath})'
    print(key, val)
    replacement_rules[key] = val

titles = {
    '1d_gp': '1D Gaussian process',
    'linear_ode': 'Linear ordinary differential equation',
    'linear_ode_ic': 'Linear ordinary differential equation with unknown initial conditions',
    'linear_ode_ic_mat': 'Linear ordinary differential equation with unknown initial conditions and ODE matrix',
    'sir': 'Toy compartmental model'
}
CommentedCodeFile.replacement_rules = replacement_rules
for name in sys.argv[1:]:
    print(name)
    CommentedExample(
        name,
        title=titles[name]
    ).write()
# exit()
# print(CommentedExample(
#     '1d_gp'
# ).markdown_content)
# exit()
# test = CommentedPythonFile(
#     'examples/1d_gp.py',
#     replacement_rules=replacement_rules,
#     figure=File(
#         pathlib.Path(*sorted(
#             pathlib.Path('examples/figs/1d_gp').glob('*/*.png'),
#             key=lambda path: path.stat().st_mtime
#         )[-1].parts[1:]),
#         description=''
#         # description=' '.join([
#         #     pathlib.Path(*sorted(
#         #         config_path.glob('*.png'),
#         #         key=lambda path: path.stat().st_mtime
#         #     )[-1].parts[1:]
#         #     for config_path in pathlib.Path('examples/figs/1d_gp').glob('*').iterdir()
#         # ])
#     )
# )
# test.markdown_file.write()
