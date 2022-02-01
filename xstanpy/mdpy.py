from xstanpy.base import *

class File(Object):
    arg_names = ('path', )
    def post_init(self):
        self.path = pathlib.Path(self.path)

    @cproperty
    def content(self):
        with open(self.path, 'r') as fd:
            return fd.read()

    @cproperty
    def lines(self): return self.content.splitlines()

    @cproperty
    def blocks(self):
        rv = [[]]
        for line in self.lines:
            if line.strip():
                rv[-1].append(line)
            elif rv[-1]:
                if rv[-1][-1].strip():
                    rv[-1].append(line)
                else:
                    rv[-1] = rv[-1][:-1]
                    rv.append([])
        return rv if rv[-1] else rv[:-1]

    def write(self, content=None):
        with open(self.path, 'w') as fd:
            return fd.write(content if content is not None else self.content)


class CommentedCodeBlock(Object):
    arg_names = ('md', 'code', 'fence')

    @cproperty
    def markdown_content(self):
        md, code = [], []
        if self.md:
            md = [self.md]
        if self.code:
            code = [f'```{self.fence}', self.code, '```']
        return '\n'.join(md+code)



class CommentedCodeFile(File):
    replacement_rules = dict()
    @cproperty
    def fence(self): return self.suffix[1:]

    inline_comment_start = '#'
    @cproperty
    def ics(self): return self.inline_comment_start
    @cproperty
    def inline_block_start(self): return f'{self.ics}~'
    @cproperty
    def ibs(self): return self.inline_block_start


    @cproperty
    def header(self):
        return f'### Commented `{self.suffix}` code ([`{self.path}`]({self.fence}/{self.path.name}))'

    @cproperty
    def blocks(self):
        rv = super().blocks
        rv[0] = [f'{self.ics} {self.header}'] + [f'{self.ics} {line}' for line in rv[0][1:-1]]
        # rv = rv[:1] + [['#', f'### Commented code ([`{self.path}`]({self.path.name}))']] + rv[1:]
        i = 0
        while i < len(rv):
            block = rv[i]
            for j, line in enumerate(block):
                line = line.strip()
                if line.startswith(self.ibs):
                    rv[i] = block[:j]
                    rv = rv[:i+1] + [block[j:]] + rv[i+1:]
                    rv[i+1][0] = self.ics + line[len(self.ibs):]
                    break
            i += 1
        return rv

    def markdown_process(self, row):
        for key, value in self.replacement_rules.items():
            row = row.replace(key, value)
        return row

    @cproperty
    def markdown_blocks(self):
        rv = []
        for block in self.blocks:
            md = []
            code = []
            for i, line in enumerate(block):
                if line.lstrip().startswith(self.ics):
                    md.append(line.lstrip()[len(self.ics):].lstrip())
                else:
                    code = block[i:]
                    break
            if len(md) == 1: md, code = [], block
            rv.append(CommentedCodeBlock(
                self.markdown_process('\n'.join(md)), '\n'.join(code), self.fence
            ).markdown_content)

            # md = [
            #     self.markdown_process(row) for row in md
            # ]
            # if len(py): py = ['```python']+py+['```']
            # rv.append('\n'.join(md+py))
        return tuple(rv)

    @cproperty
    def markdown_content(self):
        return '\n\n'.join(self.markdown_blocks)



class CommentedPythonFile(CommentedCodeFile):
    suffix = '.py'

    # @cproperty
    # def markdown_path(self): return self.path.with_suffix('.md')
#
#     @cproperty
#     def fig_block(self):
#         return f"""### Sample visualization (`config={self.figure.path.parts[-2]}`)
# # {self.figure.description}
# # ![]({self.figure.path})""".splitlines()
#
#
#     @cproperty
#     def blocks(self):
#         rv = super().blocks
#         rv[0] = ['#'+line for line in rv[0][1:-1]]
#         rv = rv[:1] + [['#', f'### Commented code ([`{self.path}`]({self.path.name}))']] + rv[1:]
#         if hasattr(self, 'figure'):
#             rv = rv[:1] + [self.fig_block] + rv[1:]
#         # rv[0] = rv[0][1:-1] + ['', f'## Commented code ([`{self.path}`]({self.path.name}))']
#         # rv[0] = ['#'+line for line in rv[0]]
#         i = 0
#         while i < len(rv):
#             block = rv[i]
#             for j, line in enumerate(block):
#                 line = line.strip()
#                 if line.startswith('#~'):
#                     rv[i] = block[:j]
#                     rv = rv[:i+1] + [block[j:]] + rv[i+1:]
#                     rv[i+1][0] = '#' + line[2:]
#                     break
#             i += 1
#         return rv
#
#     def markdown_process(self, row):
#         for key, value in self.replacement_rules.items():
#             row = row.replace(key, value)
#         return row
#
#     @cproperty
#     def markdown_blocks(self):
#         rv = []
#         for block in self.blocks:
#             md = []
#             py = []
#             for i, line in enumerate(block):
#                 if line.lstrip().startswith('#'):
#                     md.append(line.lstrip()[1:].lstrip())
#                 else:
#                     py = block[i:]
#                     break
#             if len(md) == 1: md, py = [], block
#             md = [
#                 self.markdown_process(row) for row in md
#             ]
#             if len(py): py = ['```python']+py+['```']
#             rv.append('\n'.join(md+py))
#         return tuple(rv)
#     @cproperty
#     def markdown_content(self):
#         return '\n\n'.join(self.markdown_blocks)
#     @cproperty
#     def markdown_file(self):
#         return File(self.markdown_path, content=self.markdown_content)

class CommentedStanFile(CommentedCodeFile):
    suffix = '.stan'
    inline_comment_start = '//'

class CommentedExample(Object):
    arg_names = ('name', )
    base_path = pathlib.Path('examples')
    @cproperty
    def md_path(self): return self.base_path / 'md' / f'{self.name}.md'
    @cproperty
    def py_path(self): return self.base_path / 'py' / f'{self.name}.py'
    @cproperty
    def stan_path(self): return self.base_path / 'stan' / f'{self.name}.stan'
    @cproperty
    def figs_base_path(self): return self.base_path / 'figs' / self.name
    def sorted(self, paths):
        return tuple(
            sorted(paths, key=lambda path: path.stat().st_mtime)
        )
    @cproperty
    def config_base_paths(self):
        return self.sorted([
            path
            for path in self.figs_base_path.iterdir()
            if path.is_dir()
        ])
    @cproperty
    def configuration_names(self):
        return tuple([path.name for path in self.config_base_paths])
    @cproperty
    def figure_paths(self):
        return tuple(self.figs_base_path.glob('*/*.png'))
    @cproperty
    def header_figure_path(self):
        return self.sorted(self.figure_paths)[-1]

    @cproperty
    def footer_figure_paths(self):
        return tuple([
            self.sorted(path.glob('*.png'))[-1]
            for path in self.config_base_paths
        ])
    @cproperty
    def title(self): return self.name
    @cproperty
    def markdown_header(self):
        return '\n\n'.join([
            f'![{self.header_figure_path}]({self.header_figure_path.relative_to(self.base_path)})'
        ])
    @cproperty
    def markdown_footer(self):
        return '\n\n'.join(['## Sample output'] + [
            f'### {name}\n![{path}]({path.relative_to(self.base_path)})'
            for name, path in zip(self.configuration_names, self.footer_figure_paths)
        ])

    @cproperty
    def markdown_content(self):
        return '\n\n'.join([
            File(self.md_path).content,
            self.markdown_header,
            CommentedStanFile(self.stan_path).markdown_content,
            CommentedPythonFile(self.py_path).markdown_content,
            # f'## Commented Stan code (`{self.stan_path}`)',
            # self.stan_file.markdown_content,
            # f'## Commented Python code (`{self.py_path}`)',
            # self.py_file.markdown_content,
            self.markdown_footer,
        ])

    def write(self):
        return File(
            self.base_path / f'{self.name}.md',
            content=self.markdown_content
        ).write()
