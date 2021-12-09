from xstanpy.base import *

def cpp_repr(val):
    if isinstance(val, str): return f'"{val}"'
    if isinstance(val, bool): return str(val).lower()
    return str(val)


class Argument(Object):
    arg_names = ('name', 'description')
    prefix = 'arg_'
    @cproperty
    def cpp_class_name(self):
        return f'{self.prefix}{self.name}'

    @cproperty
    def cpp_guard_name(self):
        return f'CMDSTAN_ARGUMENTS_{self.cpp_class_name.upper()}_HPP'

    cpp_include_base_names = ('singleton_argument', )

    @cproperty
    def cpp_includes(self):
        return '\n'.join([
            f'#include <cmdstan/arguments/{base_name}.hpp>'
            for base_name in self.cpp_include_base_names
        ])

    cpp_attribute_names = (
        'name',
        'description',
        'validity',
        'default',
        'default_value'
    )

    @cproperty
    def cpp_constructor_assignments(self):
        return '\n'.join([
            f'    _{key} = {cpp_repr(getattr(self, key))};'
            for key in self.cpp_attribute_names
            if hasattr(self, key)
        ])

    cpp_constructor_statements = ''
    cpp_getters = ''

    @cproperty
    def cpp(self):
        return f"""
#ifndef {self.cpp_guard_name}
#define {self.cpp_guard_name}

{self.cpp_includes}

namespace cmdstan {{

class {self.cpp_class_name} : public {self.cpp_base_class_name} {{
 public:
  {self.cpp_class_name}() {{
{self.cpp_constructor_assignments}
{self.cpp_constructor_statements}
  }}
{self.cpp_getters}
}};
}}  // namespace cmdstan
#endif

"""
    def write(self, path, dry=True):
        path = pathlib.Path(path) / f'{self.cpp_class_name}.hpp'
        if dry:
            print(f'===================== BEGIN {path} =====================')
            print(self.cpp)
            print(f'===================== END {path} =====================')
            return
        print(f'Writing to {path}')
        with open(path, 'w') as fd:
            fd.write(self.cpp)

    @cproperty
    def cpp_getter(self):
        return f'''
{self.cpp_value_type} {self.name}() {{
    return dynamic_cast<{self.cpp_class_name} *>(this->arg("{self.name}"))->value();
}};'''



class CategoricalArgument(Argument):
    cpp_base_class_name = 'categorical_argument'
    @cproperty
    def cpp_value_type(self): return f'{self.cpp_class_name}&'

    @cproperty
    def cpp_include_base_names(self):
        return (self.cpp_base_class_name, ) + tuple(set([
            subargument.cpp_class_name for subargument in self.subarguments
        ]))


    @cproperty
    def subargument_class_names(self):
        return tuple([
            subargument.cpp_class_name for subargument in self.subarguments
        ])

    @cproperty
    def cpp_constructor_statements(self):
        return '\n'.join([
            f'    _subarguments.push_back(new {class_name}());'
            for class_name in self.subargument_class_names
        ])

    @cproperty
    def cpp_getter(self):
        return f'''
{self.cpp_class_name}& {self.name}() {{
    return *dynamic_cast<{self.cpp_class_name} *>(this->arg("{self.name}"));
}};'''

    @cproperty
    def cpp_getters(self):
        return '\n'.join([
            subargument.cpp_getter for subargument in self.subarguments
        ])

    @cproperty
    def cpp_initializers(self):
        return '\n'.join([
            f'{subargument.cpp_value_type} config_{subargument.name} = config.{subargument.name}();'
            for subargument in self.subarguments
        ])

    def write(self, path, dry=True):
        super().write(path, dry)
        for subargument in self.subarguments:
            subargument.write(path, dry)

class StringArgument(Argument):
    cpp_value_type = 'std::string'
    cpp_base_class_name = 'string_argument'


class FileArgument(StringArgument):
    validity = 'Path to existing file';

class BooleanArgument(Argument):
    cpp_value_type = 'bool'
    cpp_base_class_name = 'bool_argument'
    validity = '{0,1}'
    default = '0'
    default_value = True


arg_compute = CategoricalArgument(
    'compute',
    '(Re)compute various quantities.',
    subarguments=(
        FileArgument(
            'input_path',
            '(Binary) input file of (constrained) parameter values.'
        ),
        BooleanArgument(
            'input_unconstrained',
            'Input is already unconstrained.'
        ),
        FileArgument(
            'output_path',
            '(Binary) output file of computed values.',
            validity='Writeable path'
        ),
        BooleanArgument(
            'unconstrained_parameters',
            'Compute unconstrained parameter values.'
        ),
        BooleanArgument(
            'constrained_parameters',
            'Compute constrained parameter values.'
        ),
        BooleanArgument(
            'transformed_parameters',
            'Compute transformed parameters.'
        ),
        BooleanArgument(
            'generated_quantities',
            'Compute generated quantities.'
        ),
        BooleanArgument(
            'constrained_log_probability',
            'Compute the constrained log-probability density.'
        ),
        BooleanArgument(
            'constrained_log_probability_gradient',
            'Compute the gradient of the constrained log-probability density.'
        ),
        BooleanArgument(
            'unconstrained_log_probability',
            'Compute the unconstrained log-probability density.'
        ),
        BooleanArgument(
            'unconstrained_log_probability_gradient',
            'Compute the gradient of the unconstrained log-probability density.'
        ),
    )
)
